import os, json, time, uuid, logging
from typing import Optional, List
from urllib.parse import urljoin, urlparse

import jwt
import requests
from bs4 import BeautifulSoup

from dotenv import load_dotenv
from fastapi import FastAPI, Query, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import openai  # unified with your working file

# -------------------- bootstrap --------------------
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("api")

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
JWT_SECRET      = os.getenv("JWT_SECRET", "")
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "https://11ai.ellevensa.com").split(",") if o.strip()]
WRITE_TO_DB     = os.getenv("API_WRITE_TO_DB", "0") == "1"

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY")
if not JWT_SECRET:
    raise RuntimeError("Missing JWT_SECRET")

client = openai.OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI(title="Profile Generator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ---- optional DB hooks -------------------------------------------------------
try:
    from database import fetch_profile_data, insert_generated_profile  # noqa: F401
    DB_AVAILABLE = True
except Exception as e:
    log.info(f"⚠️ database module missing/unavailable: {e}")
    DB_AVAILABLE = False

# -------------------- models --------------------
class SessionIn(BaseModel):
    user_id: int
    wp_nonce: Optional[str] = None
    request_id: Optional[str] = None

class SessionOut(BaseModel):
    session_id: str
    token: str
    request_id: Optional[str] = None

class VisibleValue(BaseModel):
    id: Optional[int] = None
    organization_name: Optional[str] = None
    about_press: Optional[str] = None
    press_date: Optional[str] = None
    article: Optional[str] = None  # for WP edited text if provided

class ChatIn(BaseModel):
    session_id: str
    user_id: int
    message: str
    visible_values: List[VisibleValue] = Field(default_factory=list)
    request_id: Optional[str] = None
    token: Optional[str] = None  # fallback if auth header stripped

# -------------------- jwt helpers --------------------
def _make_jwt(session_id: str, user_id: int) -> str:
    payload = {
        "sid": session_id,
        "uid": user_id,
        "iat": int(time.time()),
        "exp": int(time.time()) + 60 * 60 * 2,  # 2 hours
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

def _resolve_token(auth: Optional[str], x_token: Optional[str], body_token: Optional[str]) -> str:
    if auth and auth.startswith("Bearer "):
        t = auth.split(" ", 1)[1].strip()
        if t:
            return t
    if x_token and x_token.strip():
        return x_token.strip()
    if body_token and str(body_token).strip():
        return str(body_token).strip()
    raise HTTPException(status_code=401, detail="Missing token")

def _verify_jwt_any(auth: Optional[str], x_token: Optional[str], body_token: Optional[str]):
    tok = _resolve_token(auth, x_token, body_token)
    try:
        # allow small skew in case clocks differ slightly
        jwt.decode(tok, JWT_SECRET, algorithms=["HS256"], leeway=60)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Expired token")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# -------------------- misc helpers --------------------
def _clip(txt: Optional[str], max_chars: int) -> str:
    if not txt:
        return ""
    txt = txt.strip()
    return txt if len(txt) <= max_chars else txt[:max_chars] + "…"

def _values_to_context(values: List[VisibleValue]) -> str:
    if not values:
        return "لا توجد بيانات مرئية حالياً لهذا المستخدم."
    v = values[0]
    parts = []
    if v.organization_name: parts.append(f"اسم المنظمة: {v.organization_name}")
    if v.about_press:       parts.append(f"عن البيان: {v.about_press}")
    if v.press_date:        parts.append(f"تاريخ البيان: {v.press_date}")
    if v.article:           parts.append(f"المحتوى الحالي (مختصر):\n{_clip(v.article, 6000)}")
    return " | ".join(parts) if parts else "لا توجد تفاصيل كافية."

# -------------------- middleware --------------------
@app.middleware("http")
async def add_request_id_and_log(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or request.query_params.get("request_id")
    start = time.time()
    response = await call_next(request)
    if rid:
        response.headers["X-Request-ID"] = str(rid)
    dur_ms = int((time.time() - start) * 1000)
    path_q = f"{request.url.path}?{request.url.query}" if request.url.query else request.url.path
    log.info(f"[{rid or '-'}] {request.method} {path_q} -> {response.status_code} in {dur_ms}ms")
    return response

# -------------------- root / health --------------------
@app.get("/")
def root():
    return {"ok": True, "service": "profile-generator"}

@app.get("/health")
def health():
    return {"ok": True}

# -------------------- extraction --------------------
SKIP_SUBSTRINGS = (
    "/category/", "/tag/", "/feed", "/wp-json", "/author/",
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".pdf", ".zip", ".mp4", ".mp3",
)

def _same_host(a: str, b: str) -> bool:
    return urlparse(a).netloc == urlparse(b).netloc

def _should_visit(base: str, url: str) -> bool:
    if not _same_host(base, url):
        return False
    u = url.lower()
    return not any(s in u for s in SKIP_SUBSTRINGS)

def extract_info_from_url_and_subpages(base_url: str, max_pages: int = 7) -> str:
    visited, to_visit, chunks = set(), [base_url], []
    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        try:
            r = requests.get(url, timeout=12)
            r.raise_for_status()
            visited.add(url)
            soup = BeautifulSoup(r.text, "html.parser")

            page = []
            title = soup.title.string.strip() if soup.title and soup.title.string else ""
            if title: page.append(f"Title: {title}")
            desc = soup.find("meta", attrs={"name": "description"})
            if desc and desc.get("content"): page.append(f"Description: {desc['content'].strip()}")

            cnt = 0
            for p in soup.find_all("p"):
                t = p.get_text(strip=True)
                if len(t) > 50:
                    page.append(t); cnt += 1
                if cnt >= 2: break
            if page: chunks.append("\n".join(page))

            for a in soup.find_all("a", href=True):
                j = urljoin(base_url, a["href"])
                if j not in visited and j not in to_visit and _should_visit(base_url, j):
                    to_visit.append(j)
        except Exception as e:
            log.info(f"❌ Error visiting {url}: {e}")
            continue
    return "\n\n---\n\n".join(chunks)

# -------------------- OpenAI call --------------------
def call_openai_api_with_retry(examples, data: str, retries: int = 3, backoff: int = 5):
    examples_text = "\n\n".join(examples[:2])
    prompt = f"""أنت خبير محترف في إعداد الملفات التعريفية للشركات (Company Profiles)...
أمثلة:
{examples_text}

بيانات من الموقع (URL):
{data}

اكتب ملفًا تعريفياً متكاملاً (من نحن/الرؤية/الرسالة/ما الذي نقدمه/لماذا نحن/أعمالنا/خدماتنا/أسلوبنا/معلومات التواصل) بلغة عربية مؤسسية واضحة ومحترفة، مع عناوين فرعية.
"""
    for i in range(retries):
        try:
            return client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
            )
        except getattr(openai, "RateLimitError", Exception):
            if i < retries - 1:
                wait = backoff * (i + 1)
                log.info(f"Rate limit/transient error → retry in {wait}s")
                time.sleep(wait)
            else:
                raise HTTPException(status_code=429, detail="Rate limit exceeded, please try again later.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

def load_examples_from_json(path="example_profiles.json"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return [x if isinstance(x, str) else json.dumps(x, ensure_ascii=False) for x in data]
            return [json.dumps(data, ensure_ascii=False)]
    except Exception as e:
        log.info(f"⚠️ examples file issue: {e} — using defaults")
        return [
            "شركة ألفا — من نحن، الرؤية، الرسالة، خدمات أساسية، لماذا نحن، معلومات التواصل.",
            "شركة بيتا — من نحن، رؤيتنا، رسالتنا، حلول متقدمة، أسلوب العمل، بيانات الاتصال."
        ]

# -------------------- generator endpoint --------------------
@app.get("/profile-url/{user_id}/")
def profile_from_url(
    user_id: int,
    url: str = Query(..., description="Company website URL"),
    request_id: Optional[str] = Query(None),
    x_request_id: Optional[str] = Header(None),
):
    if not url or not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Invalid URL")

    rid = x_request_id or request_id
    try:
        extracted = extract_info_from_url_and_subpages(url)
        examples  = load_examples_from_json()
        resp      = call_openai_api_with_retry(examples, extracted)
        generated_profile = resp.choices[0].message.content

        if WRITE_TO_DB and DB_AVAILABLE:
            try:
                insert_generated_profile(
                    user_id=user_id,
                    organization_name=None,
                    generated_profile=generated_profile,
                    input_type="Using URL",
                    request_id=rid
                )
            except Exception as e:
                log.info(f"⚠️ insert_generated_profile failed (non-fatal): {e}")

        return {"profile": generated_profile, "request_id": rid}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"server error: {e}")

# -------------------- session & chat --------------------
@app.post("/session", response_model=SessionOut)
def create_session(body: SessionIn, x_request_id: Optional[str] = Header(None)):
    rid = x_request_id or body.request_id
    sid = str(uuid.uuid4())
    token = _make_jwt(sid, body.user_id)
    return SessionOut(session_id=sid, token=token, request_id=rid)

@app.post("/chat")
def chat(
    body: ChatIn,
    authorization: Optional[str] = Header(None),
    x_session_token: Optional[str] = Header(None),
    x_request_id: Optional[str] = Header(None),
):
    _verify_jwt_any(authorization, x_session_token, body.token)

    rid = x_request_id or body.request_id
    context = _values_to_context(body.visible_values)
    sys_prompt = (
        "أنت مساعد موثوق يجيب بدقة بالاعتماد على البيانات المرئية الحالية للمستخدم. "
        "إذا كانت المعلومة غير متوفرة في البيانات المرئية فاذكر ذلك صراحةً "
        "واقترح خطوات عملية للحصول عليها.\n\n"
        f"(RID={rid})\n"
        f"البيانات المرئية الحالية:\n{context}"
    )
    user_msg = body.message or ""

    def stream():
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user",   "content": user_msg},
            ],
            stream=True
        )
        for chunk in response:
            try:
                delta = chunk.choices[0].delta.get("content") if chunk.choices else None
            except Exception:
                delta = None
            if delta:
                yield delta

    return StreamingResponse(stream(), media_type="text/plain")
