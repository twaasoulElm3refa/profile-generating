# api.py
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

import openai  # same style as your working code

# ───────────────────────── Bootstrap ─────────────────────────
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("api")

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
JWT_SECRET      = os.getenv("JWT_SECRET", "")
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
WRITE_TO_DB     = os.getenv("API_WRITE_TO_DB", "0") == "1"   # OFF by default (avoid duplicate writes)

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
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional API-side DB hooks (kept behind WRITE_TO_DB)
try:
    from database import insert_generated_profile  # noqa: F401
    DB_AVAILABLE = True
except Exception as e:
    log.info(f"⚠️ database module missing/unavailable: {e}")
    DB_AVAILABLE = False

# ───────────────────────── Models ─────────────────────────
class SessionIn(BaseModel):
    user_id: int
    wp_nonce: Optional[str] = None
    request_id: Optional[int] = None  # WP request row id

class SessionOut(BaseModel):
    session_id: str
    token: str
    request_id: Optional[int] = None

# What the WP plugin sends today (id + article). Extras are optional/future-safe.
class VisibleValue(BaseModel):
    id: Optional[int] = None
    article: Optional[str] = None               # current plugin field
    generated_profile: Optional[str] = None     # in case you ever rename
    request_id: Optional[int] = None            # if you start sending it from WP
    website: Optional[str] = None               # if you start sending it from WP

class ChatIn(BaseModel):
    session_id: str
    user_id: int
    message: str
    visible_values: List[VisibleValue] = Field(default_factory=list)
    request_id: Optional[int] = None
    token: Optional[str] = None  # fallback if Authorization header is stripped

# ───────────────────────── JWT helpers ─────────────────────────
def _make_jwt(session_id: str, user_id: int) -> str:
    payload = {
        "sid": session_id,
        "uid": user_id,
        "iat": int(time.time()),
        "exp": int(time.time()) + 60 * 60 * 2,  # 2 hours
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

def _resolve_token(auth: Optional[str], x_token: Optional[str], body_token: Optional[str]) -> str:
    """
    Accept token from:
      - Authorization: Bearer <token>
      - X-Session-Token: <token>
      - body.token
    """
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
        jwt.decode(tok, JWT_SECRET, algorithms=["HS256"], leeway=30)  # small clock skew
    except jwt.ExpiredSignatureError:
        log.info(f"Auth failed: {type(e).__name__} – {getattr(e, 'args', [''])[0]}")
        raise HTTPException(status_code=401, detail="Expired token")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# ───────────────────────── Context helpers ─────────────────────────
def _clip(txt: Optional[str], max_chars: int) -> str:
    if not txt:
        return ""
    txt = txt.strip()
    return txt if len(txt) <= max_chars else txt[:max_chars] + "…"

def _values_to_context(values: List[VisibleValue]) -> str:
    """
    Build assistant context from WP "visible_values".
    Today WP sends {id, article}. If later you include request_id/website,
    they'll appear automatically.
    """
    if not values:
        return "لا توجد بيانات مرئية حالياً لهذا المستخدم."
    v = values[0]
    lines = []
    if v.request_id:
        lines.append(f"معرّف الطلب (RID): {v.request_id}")
    if v.website:
        lines.append(f"الموقع: {v.website}")
    # prefer article, else generated_profile
    text = v.article or v.generated_profile
    if text:
        lines.append("المحتوى الحالي (مختصر):")
        lines.append(_clip(text, 6000))
    if not lines:
        lines.append("لا توجد تفاصيل كافية.")
    return "\n".join(lines)

# ───────────────────────── Middleware ─────────────────────────
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

# ───────────────────────── Root / Health ─────────────────────────
@app.get("/")
def root():
    return {"ok": True, "service": "profile-generator"}

@app.get("/health")
def health():
    return {"ok": True}

# ───────────────────────── Web extraction ─────────────────────────
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
            r = requests.get(url, timeout=(8,12))
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

# ───────────────────────── OpenAI call ─────────────────────────
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

# ───────────────────────── Generator endpoint ─────────────────────────
@app.get("/profile-url/{user_id}/")
def profile_from_url(
    user_id: int,
    url: str = Query(..., description="Company website URL"),
    request_id: Optional[int] = Query(None),
    max_pages: int = Query(4, ge=1, le=10),      
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

        # Avoid double-writes: WP already saves in wpl3_profile_result.
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

# ───────────────────────── Session & Chat ─────────────────────────
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
    # Accept token from Authorization / X-Session-Token / body.token
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
        # Streaming text/plain so your JS uses the streaming path
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user",   "content": user_msg}
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


