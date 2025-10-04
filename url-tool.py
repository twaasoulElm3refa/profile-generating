import os
import json
import time
import uuid
import logging
from urllib.parse import urljoin, urlparse

import jwt
import requests
from bs4 import BeautifulSoup
from typing import Optional, List

from dotenv import load_dotenv
from fastapi import FastAPI, Query, HTTPException, Header, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from openai import OpenAI
import openai  # keep for installed exception classes

# -------------------- App init / config --------------------
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("api")

app = FastAPI()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set")
client = OpenAI(api_key=api_key)

JWT_SECRET = os.getenv("JWT_SECRET", "CHANGE_THIS_IN_PROD")
WRITE_TO_DB = os.getenv("API_WRITE_TO_DB", "0") == "1"
WP_ORIGIN = os.getenv("WP_ORIGIN", "https://11ai.ellevensa.com")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[WP_ORIGIN],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ---- DB hooks (your code) ----------------------------------------------------
try:
    from database import fetch_profile_data, insert_generated_profile  # noqa: F401
    DB_AVAILABLE = True
except Exception as e:
    log.info(f"⚠️ database module missing/unavailable: {e}")
    DB_AVAILABLE = False

# -------------------- Models --------------------
class SessionIn(BaseModel):
    user_id: int
    wp_nonce: Optional[str] = None
    request_id: Optional[str] = None  # WP row id

class SessionOut(BaseModel):
    session_id: str
    token: str
    request_id: Optional[str] = None

class VisibleValue(BaseModel):
    id: Optional[int] = None
    organization_name: Optional[str] = None
    about_press: Optional[str] = None
    press_date: Optional[str] = None
    article: Optional[str] = None  # WP edited content

class ChatIn(BaseModel):
    session_id: str
    user_id: int
    message: str
    visible_values: List[VisibleValue] = Field(default_factory=list)
    request_id: Optional[str] = None
    token: Optional[str] = None  # fallback if auth header is stripped

# -------------------- Helpers --------------------
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
        tok = auth.split(" ", 1)[1].strip()
        if tok:
            return tok
    if x_token and x_token.strip():
        return x_token.strip()
    if body_token and str(body_token).strip():
        return str(body_token).strip()
    raise HTTPException(status_code=401, detail="Missing token")

def _verify_jwt_any(auth: Optional[str], x_token: Optional[str], body_token: Optional[str]):
    tok = _resolve_token(auth, x_token, body_token)
    try:
        # Allow small clock skew
        jwt.decode(tok, JWT_SECRET, algorithms=["HS256"], leeway=30)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Expired token")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def _clip(txt: Optional[str], max_chars: int) -> str:
    if txt is None:
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

# -------------------- Middleware (Request-ID + logging) --------------------
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

# -------------------- Simple root & health --------------------
@app.get("/")
def root():
    return {"ok": True, "service": "profile-generator", "origin": WP_ORIGIN}

@app.get("/health")
def health():
    return {"ok": True}

# -------------------- OpenAI call --------------------
def call_openai_api_with_retry(examples, data: str, retries: int = 3, backoff: int = 5):
    examples_text = "\n\n".join(examples[:2])  # keep short
    prompt = f"""أنت خبير محترف في إعداد الملفات التعريفية للشركات (Company Profiles)...
أمثلة:
{examples_text}

بيانات من الموقع (URL):
{data}

اكتب ملفًا تعريفياً متكاملاً (من نحن/الرؤية/الرسالة/ما الذي نقدمه/لماذا نحن/أعمالنا/خدماتنا/أسلوبنا/معلومات التواصل) بلغة عربية مؤسسية واضحة ومحترفة، مع عناوين فرعية.
"""
    for i in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                # temperature omitted → avoids “only default (1) supported” surprises
            )
            return response
        except getattr(openai, "RateLimitError", Exception):
            if i < retries - 1:
                wait_time = backoff * (i + 1)
                log.info(f"Rate limit/transient error. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise HTTPException(status_code=429, detail="Rate limit exceeded, please try again later.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

# -------------------- Extraction --------------------
SKIP_SUBSTRINGS = (
    "/category/", "/tag/", "/feed", "/wp-json", "/author/",
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".pdf", ".zip", ".mp4", ".mp3",
)

def should_visit(base: str, url: str) -> bool:
    if any(s in url.lower() for s in SKIP_SUBSTRINGS):
        return False
    # stay on same host
    return urlparse(base).netloc == urlparse(url).netloc

def extract_info_from_url_and_subpages(base_url, max_pages=7):
    visited = set()
    to_visit = [base_url]
    all_texts = []

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        try:
            resp = requests.get(url, timeout=12)
            resp.raise_for_status()
            html = resp.text
            soup = BeautifulSoup(html, "html.parser")
            visited.add(url)

            page_text = []
            title = soup.title.string.strip() if soup.title and soup.title.string else ""
            if title:
                page_text.append(f"Title: {title}")

            desc_tag = soup.find("meta", attrs={"name": "description"})
            if desc_tag and desc_tag.get("content"):
                page_text.append(f"Description: {desc_tag['content'].strip()}")

            # a couple of meaningful paragraphs
            count = 0
            for p in soup.find_all("p"):
                text = p.get_text(strip=True)
                if len(text) > 50:
                    page_text.append(text)
                    count += 1
                if count >= 2:
                    break

            if page_text:
                all_texts.append("\n".join(page_text))

            for link in soup.find_all("a", href=True):
                joined = urljoin(base_url, link["href"])
                if joined not in visited and joined not in to_visit and should_visit(base_url, joined):
                    to_visit.append(joined)

        except Exception as e:
            log.info(f"❌ Error visiting {url}: {e}")
            continue

    return "\n\n---\n\n".join(all_texts)

# -------------------- Examples loader (safe) --------------------
def load_examples_from_json(json_path="example_profiles.json"):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
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

# -------------------- Generator endpoint --------------------
@app.get("/profile-url/{user_id}/")
def profile_from_url(
    user_id: int,
    url: str = Query(..., description="Company website URL"),
    request_id: Optional[str] = Query(None, description="WP request id (wpl3_profile_generating_tool.id)"),
    x_request_id: Optional[str] = Header(None),
):
    if not url or not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Invalid URL")

    rid = x_request_id or request_id

    try:
        extracted_data  = extract_info_from_url_and_subpages(url)
        loaded_examples = load_examples_from_json()

        response = call_openai_api_with_retry(loaded_examples, extracted_data)
        generated_profile = response.choices[0].message.content

        # Optional API-side DB write (defaults OFF)
        if WRITE_TO_DB and DB_AVAILABLE:
            try:
                insert_generated_profile(
                    user_id=user_id,
                    organization_name=None,
                    generated_profile=generated_profile,
                    input_type='Using URL',
                    request_id=rid
                )
            except Exception as db_e:
                log.info(f"⚠️ insert_generated_profile failed (non-fatal): {db_e}")

        return {"profile": generated_profile, "request_id": rid}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"server error: {e}")

# -------------------- Session & Chat --------------------
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
    x_request_id: Optional[str] = Header(None)
):
    # accept token from multiple places
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
                {"role": "user",   "content": user_msg}
            ],
            stream=True  # temperature omitted
        )
        for chunk in response:
            try:
                delta = chunk.choices[0].delta.get("content") if chunk.choices else None
            except Exception:
                delta = None
            if delta:
                yield delta

    return StreamingResponse(stream(), media_type="text/plain")
