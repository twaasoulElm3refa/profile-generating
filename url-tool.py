import os
import json
import time
import uuid
from typing import Optional, List

import jwt  # PyJWT
import requests
from urllib.parse import urljoin, urlparse

from dotenv import load_dotenv
from bs4 import BeautifulSoup

from fastapi import FastAPI, Query, HTTPException, Header, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from openai import OpenAI
import openai  # keep for RateLimitError / compat

# -------------------- App init / config --------------------
load_dotenv()
app = FastAPI()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set")
client = OpenAI(api_key=api_key)

# IMPORTANT: All API instances MUST use the same secret
JWT_SECRET = os.getenv("JWT_SECRET", "CHANGE_THIS_IN_PROD")

# Update origins to include your WP domain(s)
origins = [
    "https://11ai.ellevensa.com",  # WordPress site
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ---- DB hooks (optional, safe fallback) -------------------
try:
    from database import fetch_profile_data, insert_generated_profile  # type: ignore
except Exception as _db_e:
    print(f"⚠️ database.py not available ({_db_e}); using no-op hooks.")

    def fetch_profile_data(*args, **kwargs):
        return None

    def insert_generated_profile(
        user_id: int,
        organization_name: Optional[str],
        generated_profile: str,
        input_type: str = "Using URL",
        request_id: Optional[str] = None,
    ):
        # no-op in dev
        return None


# -------------------- Models --------------------
class SessionIn(BaseModel):
    user_id: int
    wp_nonce: Optional[str] = None
    request_id: Optional[str] = None  # carries WP request_id (the row id)

class SessionOut(BaseModel):
    session_id: str
    token: str
    request_id: Optional[str] = None

class VisibleValue(BaseModel):
    id: Optional[int] = None
    organization_name: Optional[str] = None
    about_press: Optional[str] = None
    press_date: Optional[str] = None
    article: Optional[str] = None  # used by the WP plugin

class ChatIn(BaseModel):
    session_id: str
    user_id: int
    message: str
    visible_values: List[VisibleValue] = Field(default_factory=list)
    request_id: Optional[str] = None  # for tracing a specific chat turn
    token: Optional[str] = None       # <-- NEW: allow token in body as fallback


# -------------------- Helpers --------------------
def _make_jwt(session_id: str, user_id: int) -> str:
    payload = {
        "sid": session_id,
        "uid": user_id,
        "iat": int(time.time()),
        "exp": int(time.time()) + 60 * 60 * 2,  # 2 hours
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

def _extract_token(authorization: Optional[str], x_session_token: Optional[str], body_token: Optional[str]) -> str:
    """
    Accept token from:
      - Authorization: Bearer <token>
      - X-Session-Token: <token>
      - request body: token
    Trims quotes/whitespace and 'Bearer ' prefix.
    """
    cands = []
    if authorization:
        cands.append(authorization.strip())
    if x_session_token:
        cands.append(x_session_token.strip())
    if body_token:
        cands.append(str(body_token).strip())

    for c in cands:
        if not c:
            continue
        if c.lower().startswith("bearer "):
            c = c.split(" ", 1)[1]
        c = c.strip().strip('"').strip("'")
        if c:
            return c
    return ""

def _verify_token_any(authorization: Optional[str], x_session_token: Optional[str], body_token: Optional[str]):
    token = _extract_token(authorization, x_session_token, body_token)
    if not token:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    try:
        # accept small clock skew
        jwt.decode(token, JWT_SECRET, algorithms=["HS256"], options={"leeway": 60})
    except jwt.InvalidTokenError:
        # match your previous observable error message
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
    print(f"[{rid or '-'}] {request.method} {path_q} -> {response.status_code} in {dur_ms}ms")
    return response


# -------------------- OpenAI calls --------------------
def call_openai_api_with_retry(examples, data: str, retries: int = 3, backoff: int = 5):
    """
    Used by /profile-url (non-stream). Includes fallback if 'temperature' is not supported.
    """
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
            try:
                # First try with temperature
                return client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                )
            except Exception as e1:
                # Some orgs/models only accept default temperature
                return client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                )
        except getattr(openai, "RateLimitError", Exception) as e:
            if i < retries - 1:
                wait_time = backoff * (i + 1)
                print(f"Rate limit/transient error. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise HTTPException(status_code=429, detail="Rate limit exceeded, please try again later.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


# -------------------- Extraction --------------------
def extract_info_from_url_and_subpages(base_url, max_pages=7):
    visited = set()
    to_visit = [base_url]
    all_texts = []

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            html = response.text
            soup = BeautifulSoup(html, "html.parser")
            visited.add(url)

            page_text = []
            title = soup.title.string.strip() if soup.title and soup.title.string else ""
            if title:
                page_text.append(f"Title: {title}")

            desc_tag = soup.find("meta", attrs={"name": "description"})
            if desc_tag and desc_tag.get("content"):
                page_text.append(f"Description: {desc_tag['content'].strip()}")

            paragraphs = soup.find_all("p")
            count = 0
            for p in paragraphs:
                text = p.get_text(strip=True)
                if len(text) > 50:
                    page_text.append(text)
                    count += 1
                if count >= 2:
                    break

            if page_text:
                all_texts.append("\n".join(page_text))

            for link in soup.find_all("a", href=True):
                href = link["href"]
                joined_url = urljoin(base_url, href)
                parsed_base = urlparse(base_url)
                parsed_joined = urlparse(joined_url)
                if parsed_base.netloc == parsed_joined.netloc:
                    if joined_url not in visited and joined_url not in to_visit:
                        to_visit.append(joined_url)

        except Exception as e:
            print(f"❌ Error visiting {url}: {e}")
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
        # Safe defaults if file missing
        print(f"⚠️ examples file issue: {e} — using defaults")
        return [
            "شركة ألفا — من نحن، الرؤية، الرسالة، خدمات أساسية، لماذا نحن، معلومات التواصل.",
            "شركة بيتا — من نحن، رؤيتنا، رسالتنا، حلول متقدمة، أسلوب العمل، بيانات الاتصال."
        ]


# -------------------- Health --------------------
@app.get("/health")
def health():
    return {"ok": True}


# -------------------- Generator endpoint --------------------
@app.get("/profile-url/{user_id}/")
def profile_from_url(
    user_id: int,
    url: str = Query(..., description="Company website URL"),
    request_id: Optional[str] = Query(None, description="WP request id (wpl3_profile_generating_tool.id)"),
):
    if not url or not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Invalid URL")

    try:
        extracted_data  = extract_info_from_url_and_subpages(url)
        loaded_examples = load_examples_from_json()

        response = call_openai_api_with_retry(loaded_examples, extracted_data)
        generated_profile = response.choices[0].message.content

        # Save to API-side DB (optional). We also forward request_id if your table has it.
        try:
            insert_generated_profile(
                user_id=user_id,
                organization_name=None,
                generated_profile=generated_profile,
                input_type='Using URL',
                request_id=request_id  # pass through for traceability
            )
        except Exception as db_e:
            print(f"⚠️ insert_generated_profile failed (non-fatal): {db_e}")

        return {"profile": generated_profile, "request_id": request_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"server error: {e}")


# -------------------- Session & Chat --------------------
@app.post("/session", response_model=SessionOut)
def create_session(body: SessionIn, x_request_id: Optional[str] = Header(None)):
    # Echo back the request id we got (prefer header, else body)
    rid = x_request_id or body.request_id
    sid = str(uuid.uuid4())
    token = _make_jwt(sid, body.user_id)
    return SessionOut(session_id=sid, token=token, request_id=rid)


@app.post("/chat")
def chat(
    body: ChatIn,
    authorization: Optional[str] = Header(None),
    x_session_token: Optional[str] = Header(None),   # <-- NEW: alt header
    x_request_id: Optional[str] = Header(None)
):
    # Robust verification from any source (Authorization / X-Session-Token / body.token)
    _verify_token_any(authorization, x_session_token, body.token)

    # prefer header RID; else body.request_id
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
        """
        Stream with fallback if temperature is not supported for the current org/model.
        """
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.2,  # some orgs do accept; if not, we fallback
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user",   "content": user_msg}
                ],
                stream=True
            )
        except Exception:
            # Retry with default temperature (omit param)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user",   "content": user_msg}
                ],
                stream=True
            )

        for chunk in response:
            delta = None
            try:
                delta = chunk.choices[0].delta.get("content") if chunk.choices else None
            except Exception:
                pass
            if delta:
                yield delta

    return StreamingResponse(stream(), media_type="text/plain; charset=utf-8")
