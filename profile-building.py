# api.py
import os, json, time, uuid, logging
from typing import Optional, List
from urllib.parse import urlparse

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

import jwt
from openai import OpenAI

# Your DB helpers (as in your code)
from database import fetch_profile_data   #, insert_generated_profil

# ─────────────────── Bootstrap ───────────────────
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("pg-api")

OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "")
JWT_SECRET       = os.getenv("JWT_SECRET", "") or os.getenv("SECRET_KEY", "")
ALLOWED_ORIGINS  = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
TOKEN_TTL_SEC    = int(os.getenv("TOKEN_TTL_SEC", "1800"))  # 30 min
API_WRITE_TO_DB  = os.getenv("API_WRITE_TO_DB", "0") == "1"  # default OFF to avoid duplicate rows

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY")
if not JWT_SECRET:
    raise RuntimeError("Missing JWT_SECRET")

client = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI(title="Profile Generating Tool")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS != ["*"] else ["*"],
    allow_credentials=False,  # header auth, not cookies
    allow_methods=["GET","POST","OPTIONS"],
    allow_headers=["Authorization","Content-Type","X-Session-Token","X-Request-ID"],
    expose_headers=["X-Request-ID"],
)

# ─────────────────── Utils ───────────────────
def _same_host(a: str, b: str) -> bool:
    try:
        return urlparse(a).netloc == urlparse(b).netloc
    except Exception:
        return False

def load_examples_from_json(json_path: str = "example_profiles.json"):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return [x if isinstance(x, str) else json.dumps(x, ensure_ascii=False) for x in data]
            return [json.dumps(data, ensure_ascii=False)]
    except Exception as e:
        log.info(f"⚠ examples file issue: {e} — using defaults")
        return [
            "شركة ألفا — من نحن، الرؤية، الرسالة، خدمات أساسية، لماذا نحن، معلومات التواصل.",
            "شركة بيتا — من نحن، رؤيتنا، رسالتنا، حلول متقدمة، أسلوب العمل، بيانات الاتصال."
        ]

def _clip(txt: Optional[str], n: int) -> str:
    if not txt:
        return ""
    txt = txt.strip()
    return txt if len(txt) <= n else txt[:n] + "…"

# ─────────────────── OpenAI prompts ───────────────────
def generate_profile_text(data: dict, examples: List[str]) -> str:
    """
    data is a dict from fetch_profile_data(user_id): e.g.
    {
      'organization_name': ..., 'vision': ..., 'message': ..., 'about_organization': ...,
      'services': ..., 'phone': ..., 'website': ..., 'email': ..., 'location': ..., 'more_information': ...
    }
    """
    examples_text = "\n\n".join(examples[:2])
    # Serialize the latest row as compact JSON for the prompt
    data_json = json.dumps(data, ensure_ascii=False , default=str)

    examples_text = "\n\n".join(examples[:2])  # نرسل أول مثالين فقط لتقليل الطول
    prompt=f'''أنت خبير متخصص في كتابة الملفات التعريفية للشركات (Company Profiles)، وتعمل كمستشار استراتيجي في تطوير الهوية المؤسسية والعرض الاحترافي للخدمات.
ستتلقى مجموعة من الملفات تتضمن محتوى خام وتعريفي من عدة شركات{examples_text}، دورك هو أن تحلل هذه الملفات بدقة، وتستخلص منها الأسلوب الاحترافي الأمثل لبناء ملف تعريفي متميز ومتكامل لشركة معلوماتها فى {data_json}مع ذكر الرؤيه والرساله فى فقرات منفصله .
المطلوب:
    كتابة ملف تعريفي احترافي للشركة بأسلوب عصري وجذاب، يُراعي اللغة المؤسسية، ويُبرز الهوية والمكانة التنافسية.
    لا تعتمد على هيكل جاهز، بل ابتكر ترتيبًا منطقيًا وتدريجيًا للمحتوى يُناسب الشركة ومجالها.
    اجعل الملف غنيًا بالتفاصيل، ومتوازنًا بين النصوص التسويقية، والمحتوى المعلوماتي، والمزايا التنافسية.
    ضمّن أقسامًا مثل: من نحن، ما الذي نُقدمه، لماذا نحن، أعمالنا، خدماتنا، أسلوبنا، وغيرها إن وجدت مناسبة.
    اجعل الكتابة سلسة، متماسكة، ومتناسقة بصريًا ومضمونيًا.
    افترض أنك تكتب الملف ليُستخدم في طباعة فاخرة، وعرض إلكتروني، وعرض تقديمي.
سيُستخدم الملف لاحقًا من قِبل جهات استثمارية وعملاء محتملين، لذا يجب أن يُجسّد هوية الشركة وقوتها.
'''

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return resp.choices[0].message.content

# ─────────────────── JWT helpers (for chat) ───────────────────
def _make_jwt(session_id: str, user_id: int) -> str:
    now = int(time.time())
    payload = {"sid": session_id, "uid": user_id, "iat": now, "exp": now + TOKEN_TTL_SEC}
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

def _resolve_token(auth: Optional[str], x_token: Optional[str], body_token: Optional[str]) -> str:
    if auth and auth.startswith("Bearer "):
        t = auth.split(" ", 1)[1].strip()
        if t: return t
    if x_token and x_token.strip():
        return x_token.strip()
    if body_token and str(body_token).strip():
        return str(body_token).strip()
    raise HTTPException(status_code=401, detail="Missing token")

def _verify_jwt_any(auth: Optional[str], x_token: Optional[str], body_token: Optional[str]):
    tok = _resolve_token(auth, x_token, body_token)
    try:
        jwt.decode(tok, JWT_SECRET, algorithms=["HS256"], leeway=30)
    except jwt.ExpiredSignatureError as e:
        log.info(f"Auth failed: {type(e).__name__} – {e}")
        raise HTTPException(status_code=401, detail="Expired token")
    except jwt.InvalidTokenError as e:
        log.info(f"Auth failed: {type(e).__name__} – {e}")
        raise HTTPException(status_code=401, detail="Invalid token")

# ─────────────────── Middleware: echo request id ───────────────────
@app.middleware("http")
async def add_request_id_header(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or request.query_params.get("request_id")
    start = time.time()
    try:
        response = await call_next(request)
    except Exception as e:
        # ensure rid is present even on errors
        response = JSONResponse({"detail": str(e)}, status_code=500)
    if rid:
        response.headers["X-Request-ID"] = str(rid)
    dur = int((time.time() - start) * 1000)
    log.info(f"[{rid or '-'}] {request.method} {request.url.path} -> {response.status_code} in {dur}ms")
    return response

# ─────────────────── Health ───────────────────
@app.get("/health")
def health():
    return {"ok": True}

# ─────────────────── Generator endpoint ───────────────────
@app.get("/profile-generating-tool/{user_id}")
def profile_generating_tool(
    user_id: int,
    request_id: Optional[int] = Query(None, description="Optional request id"),
    x_request_id: Optional[str] = Header(None),
):
    """
    Used by WP plugin to fetch the generated profile for the latest form data row.
    It returns JSON: { "generated_profile": "..." }
    The plugin itself will store into wpl3_profile_result with input_type='Using FORM'.
    """
    rid = x_request_id or request_id
    try:
        rows = fetch_profile_data(user_id)
        if not rows:
            raise HTTPException(status_code=404, detail="لا توجد بيانات للمستخدم.")
        data = rows[-1]

        loaded_examples = load_examples_from_json()
        generated_profile = generate_profile_text(data, loaded_examples)
        #save_data= insert_generated_profile(user_id,data['organization_name'],generated_profile)

        # Avoid duplicate writes: plugin is already saving the result row
        '''if API_WRITE_TO_DB:
            try:
                insert_generated_profile(
                    user_id,
                    data.get("organization_name") or None,
                    generated_profile
                )
            except Exception as e:
                log.info(f"⚠ insert_generated_profile failed (non-fatal): {e}")'''

        resp = JSONResponse(content={"generated_profile": generated_profile})
        if rid:
            resp.headers["X-Request-ID"] = str(rid)
        return resp

    except HTTPException:
        raise
    except Exception as e:
        log.exception("generator error")
        resp = JSONResponse(content={"error": str(e)}, status_code=500)
        if rid:
            resp.headers["X-Request-ID"] = str(rid)
        return resp

# ─────────────────── Chat models ───────────────────
from pydantic import BaseModel, Field

class VisibleValue(BaseModel):
    id: Optional[int] = None
    article: Optional[str] = None
    request_id: Optional[int] = None
    website: Optional[str] = None

class SessionIn(BaseModel):
    user_id: int
    request_id: Optional[int] = None

class SessionOut(BaseModel):
    session_id: str
    token: str
    request_id: Optional[int] = None

class ChatIn(BaseModel):
    session_id: str
    user_id: int
    message: str
    visible_values: List[VisibleValue] = Field(default_factory=list)
    request_id: Optional[int] = None
    token: Optional[str] = None  # fallback if proxy strips Authorization header

# ─────────────────── Chat endpoints (under the same base path) ───────────────────
@app.post("/session", response_model=SessionOut)
def create_session(body: SessionIn, x_request_id: Optional[str] = Header(None)):
    rid = x_request_id or body.request_id
    sid = str(uuid.uuid4())
    token = _make_jwt(sid, body.user_id)
    resp = SessionOut(session_id=sid, token=token, request_id=rid)
    return resp

def _values_to_context(values: List[VisibleValue]) -> str:
    if not values:
        return "لا توجد بيانات مرئية حالياً لهذا المستخدم."
    v = values[0]
    lines = []
    if v.request_id:
        lines.append(f"معرّف الطلب (RID): {v.request_id}")
    if v.website:
        lines.append(f"الموقع: {v.website}")
    if v.article:
        lines.append("المحتوى الحالي (مختصر):")
        lines.append(_clip(v.article, 6000))
    if not lines:
        lines.append("لا توجد تفاصيل كافية.")
    return "\n".join(lines)

@app.post("/chat")
def chat(
    body: ChatIn,
    authorization: Optional[str] = Header(None),
    x_session_token: Optional[str] = Header(None),
    x_request_id: Optional[str] = Header(None),
):
    # Auth
    _verify_jwt_any(authorization, x_session_token, body.token)

    rid = x_request_id or body.request_id
    context = _values_to_context(body.visible_values)
    sys_prompt = (
        "أنت مساعد موثوق يجيب بدقة بالاعتماد على البيانات المرئية الحالية للمستخدم. "
        "إذا كانت المعلومة غير متوفرة في البيانات فاذكر ذلك صراحةً "
        "واقترح خطوات عملية للحصول عليها.\n\n"
        f"(RID={rid})\n"
        f"البيانات المرئية الحالية:\n{context}"
    )
    user_msg = body.message or ""

    def stream():
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user",   "content": user_msg},
                ],
                stream=True,
            )
            for chunk in response:
                try:
                    # OpenAI SDK (responses.create, delta.content)
                    choice = chunk.choices[0]
                    delta = getattr(choice, "delta", None)
                    content = getattr(delta, "content", None)
                    if content:
                        yield content
                except Exception:
                    # ignore malformed chunks
                    continue
        except Exception as e:
            # surface the error at the end of the stream
            yield f"\n[خطأ]: {str(e)}"

    headers = {}
    if rid:
        headers["X-Request-ID"] = str(rid)
    return StreamingResponse(stream(), media_type="text/plain", headers=headers)

