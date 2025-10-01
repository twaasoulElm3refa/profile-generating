import os
import json
import time
import uuid
import jwt
import requests
from urllib.parse import urljoin, urlparse

from dotenv import load_dotenv
from bs4 import BeautifulSoup

from typing import Optional, List
from pydantic import BaseModel, Field

from fastapi import FastAPI, Query, HTTPException, Header
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from openai import OpenAI
import openai  # keep for error classes in some installs

# ---- App init / config -------------------------------------------------------

load_dotenv()
app = FastAPI()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set")

client = OpenAI(api_key=api_key)

JWT_SECRET = os.getenv("JWT_SECRET", "CHANGE_THIS_IN_PROD")

origins = [
    "https://11ai.ellevensa.com",  # <-- your WordPress site
    # "http://localhost:8000",      # (optional) local test
    # "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---- DB hooks (your code) ----------------------------------------------------
# Keep your own implementations & imports:
from database import fetch_profile_data, insert_generated_profile  # noqa: E402

# ---- Models for chat/session -------------------------------------------------

class SessionIn(BaseModel):
    user_id: int
    wp_nonce: Optional[str] = None

class SessionOut(BaseModel):
    session_id: str
    token: str

class VisibleValue(BaseModel):
    id: Optional[int] = None
    organization_name: Optional[str] = None
    about_press: Optional[str] = None
    press_date: Optional[str] = None
    article: Optional[str] = None  # <-- used by the WP plugin

class ChatIn(BaseModel):
    session_id: str
    user_id: int
    message: str
    visible_values: List[VisibleValue] = Field(default_factory=list)

# ---- Helpers (JWT + context) -------------------------------------------------

def _make_jwt(session_id: str, user_id: int) -> str:
    payload = {
        "sid": session_id,
        "uid": user_id,
        "iat": int(time.time()),
        "exp": int(time.time()) + 60 * 60 * 2,  # 2 hours
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

def _verify_jwt(bearer: Optional[str]):
    if not bearer or not bearer.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = bearer.split(" ", 1)[1]
    try:
        jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def _clip(txt: str, max_chars: int) -> str:
    if txt is None:
        return ""
    txt = txt.strip()
    if len(txt) <= max_chars:
        return txt
    return txt[:max_chars] + "…"

def _values_to_context(values: List[VisibleValue]) -> str:
    if not values:
        return "لا توجد بيانات مرئية حالياً لهذا المستخدم."
    v = values[0]
    parts = []
    # These are optional; present only if available
    if v.organization_name:
        parts.append(f"اسم المنظمة: {v.organization_name}")
    if v.about_press:
        parts.append(f"عن البيان: {v.about_press}")
    if v.press_date:
        parts.append(f"تاريخ البيان: {v.press_date}")
    if v.article:
        # keep article reasonable in size to control tokens
        parts.append(f"المحتوى الحالي (مختصر):\n{_clip(v.article, 6000)}")
    return " | ".join(parts) if parts else "لا توجد تفاصيل كافية."

# ---- Your existing OpenAI call (kept) ---------------------------------------

def call_openai_api_with_retry(examples, data: str, retries: int = 3, backoff: int = 5):
    examples_text = "\n\n".join(examples[:2])  # نرسل أول مثالين فقط لتقليل الطول
    prompt = f""" أنت خبير محترف في إعداد الملفات التعريفية للشركات (Company Profiles)، وتعمل كمستشار استراتيجي لتطوير الهوية المؤسسية وصياغة المحتوى التسويقي الاحترافي
        ستتلقى:
        - أمثلة حقيقية لملفات تعريفية ناجحة لعدة شركات:
        {examples_text}
        
        ومعلومات أساسية تم استخراجها مباشرةً من موقع الشركة (URL):
        {data}
        ---
        
        📌 المطلوب منك:
        1️⃣ تحليل الأمثلة الواردة لاستخلاص أسلوب احترافي متكامل في كتابة الملفات التعريفية.
        2️⃣ الاستفادة من البيانات المستخرجة من الموقع (url) كما هي تمامًا، وإن لم تكن المعلومات مكتملة؛ قم بإكمالها وابتكار محتوى مكمل بأسلوب متناسق.
        3️⃣ كتابة ملف تعريفي متكامل يشمل:
           - من نحن
           - الرؤية (في فقرة منفصلة)
           - الرسالة (في فقرة منفصلة)
           - ما الذي نُقدمه
           - لماذا نحن
           - أعمالنا
           - خدماتنا (مفصلة بنقاط)
           - أسلوبنا
           - معلومات التواصل
        
        ---
        
        ✅ تعليمات أساسية:
        - استخدم أسلوب عصري وجذاب يوازن بين النص التسويقي والمعلوماتي.
        - لا تعتمد على هيكل جاهز حرفيًا؛ ابتكر ترتيبًا تدريجيًا يناسب مجال الشركة.
        - اجعل النص غنيًا بالتفاصيل ويعكس الهوية التنافسية المستخلصة من الأمثلة.
        - استخدم لغة مؤسسية سلسة ومتماسكة بصريًا ومضمونيًا.
        - افترض أن الملف سيُستخدم للطباعة الفاخرة والعروض الإلكترونية والتقديمية.
        - الملف يجب أن يُجسّد هوية الشركة ويقنع الجهات الاستثمارية والعملاء المستهدفين.
        
        ---
        
        ✨ الهدف:
        إنشاء ملف تعريفي قوي يعبر عن روح الشركة بأسلوب مستوحى ومتعلَّم من الأمثلة الواردة، مع ملء أي نقص في بيانات الموقع تلقائيًا بأسلوب احترافي.
        """
    for i in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            return response
        except getattr(openai, "RateLimitError", Exception) as e:  # compatible across lib versions
            if i < retries - 1:
                wait_time = backoff * (i + 1)
                print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise HTTPException(status_code=429, detail="Rate limit exceeded, please try again later.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

# ---- Website extraction (kept) ----------------------------------------------

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

            # collect texts
            page_text = []

            # title
            title = soup.title.string.strip() if soup.title and soup.title.string else ""
            if title:
                page_text.append(f"Title: {title}")

            # meta description
            desc_tag = soup.find("meta", attrs={"name": "description"})
            if desc_tag and desc_tag.get("content"):
                page_text.append(f"Description: {desc_tag['content'].strip()}")

            # first two long paragraphs
            paragraphs = soup.find_all("p")
            count = 0
            for p in paragraphs:
                text = p.get_text(strip=True)
                if len(text) > 50:
                    page_text.append(text)
                    count += 1
                if count >= 2:
                    break

            all_texts.append("\n".join(page_text))

            # next internal links
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

    combined_text = "\n\n---\n\n".join(all_texts)
    return combined_text

# ---- Examples loader (kept) -------------------------------------------------

def load_examples_from_json(json_path="example_profiles.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---- EXISTING ENDPOINT (kept) -----------------------------------------------

@app.get("/profile-url/{user_id}/")
def profile_from_url(user_id: int, url: str = Query(..., description="Company website URL")):
    if not url or not url.startswith("http"):
        raise HTTPException(status_code=400, detail="Invalid URL")
    print(user_id, url)

    extracted_data = extract_info_from_url_and_subpages(url)
    loaded_examples = load_examples_from_json()

    try:
        response = call_openai_api_with_retry(loaded_examples, extracted_data)
        generated_profile = response.choices[0].message.content
        input_type = 'Using URL'
        # save to DB
        save_data = insert_generated_profile(user_id, None, generated_profile, input_type)
        return {"profile": generated_profile}
    except HTTPException as e:
        raise e

# ---- NEW: Session + Chat (streaming) ----------------------------------------

@app.post("/session", response_model=SessionOut)
def create_session(body: SessionIn):
    sid = str(uuid.uuid4())
    token = _make_jwt(sid, body.user_id)
    return SessionOut(session_id=sid, token=token)

@app.post("/chat")
def chat(body: ChatIn, authorization: Optional[str] = Header(None)):
    _verify_jwt(authorization)

    context = _values_to_context(body.visible_values)
    sys_prompt = (
        "أنت مساعد موثوق يجيب بدقة بالاعتماد على البيانات المرئية الحالية للمستخدم. "
        "إذا كانت المعلومة غير متوفرة في البيانات المرئية فاذكر ذلك صراحةً "
        "واقترح خطوات عملية للحصول عليها.\n\n"
        f"البيانات المرئية الحالية:\n{context}"
    )
    user_msg = body.message or ""

    def stream():
        # Use a light, fast model for chat streaming
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
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
