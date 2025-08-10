import requests
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from fastapi import FastAPI, Query, HTTPException
from database import fetch_profile_data ,insert_generated_profile
from pydantic import BaseModel
#from typing import Optional
from fastapi.responses import FileResponse
import os
from dotenv import load_dotenv
import json
from openai import OpenAI
import time
import openai 

# تحميل متغيرات البيئة
load_dotenv()
app = FastAPI()
#api_key=os.getenv("OPENAI_API_KEY")

origins = [
    "https://11ai.ellevensa.com",  # Replace with your WordPress site domain
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows specific origins to make requests
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Allows specific methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers (or specify as needed)
)

# Function to call the OpenAI API and handle rate limit errors
def call_openai_api_with_retry(examples , data: str ,retries: int = 3, backoff: int = 5):
    client = OpenAI(api_key=api_key)
    examples_text = "\n\n".join(examples[:2])  # نرسل أول مثالين فقط لتقليل الطول
    prompt=f""" أنت خبير محترف في إعداد الملفات التعريفية للشركات (Company Profiles)، وتعمل كمستشار استراتيجي لتطوير الهوية المؤسسية وصياغة المحتوى التسويقي الاحترافي
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
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            )
            return response
        except openai.error.RateLimitError as e:
            if i < retries - 1:
                wait_time = backoff * (i + 1)  # Exponential backoff
                print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)  # Wait before retrying
            else:
                raise HTTPException(status_code=429, detail="Rate limit exceeded, please try again later.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) 

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

            # جمع النصوص
            page_text = []

            # العنوان
            title = soup.title.string.strip() if soup.title and soup.title.string else ""
            if title:
                page_text.append(f"Title: {title}")

            # meta description
            desc_tag = soup.find("meta", attrs={"name": "description"})
            if desc_tag and desc_tag.get("content"):
                page_text.append(f"Description: {desc_tag['content'].strip()}")

            # أول فقرتين طويلتين
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

            # استخراج روابط داخلية جديدة لزيارتها لاحقًا
            for link in soup.find_all("a", href=True):
                href = link["href"]
                joined_url = urljoin(base_url, href)
                parsed_base = urlparse(base_url)
                parsed_joined = urlparse(joined_url)
                # نتأكد أن الرابط داخلي (نفس الدومين)
                if parsed_base.netloc == parsed_joined.netloc:
                    if joined_url not in visited and joined_url not in to_visit:
                        to_visit.append(joined_url)

        except Exception as e:
            print(f"❌ Error visiting {url}: {e}")
            continue

    # دمج كل النصوص المستخرجة
    combined_text = "\n\n---\n\n".join(all_texts)
    return combined_text


def load_examples_from_json(json_path="example_profiles.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/profile-url/{user_id}/")
def profile_from_url(user_id: int,url: str = Query(..., description="Company website URL") ):
    #try:
    if not url or not url.startswith("http"):
        raise HTTPException(status_code=400, detail="Invalid URL")
    print (user_id, url)
    # استخرج البيانات من الرابط
    extracted_data = extract_info_from_url_and_subpages(url)

    # نقرأ الأمثلة
    loaded_examples = load_examples_from_json()

    # توليد البروفايل
    #generated_profile = generate_profile_model(extracted_data, loaded_examples)
    #print(generated_profile)

    try:
        #response = call_openai_api_with_retry(extracted_data,loaded_examples)
        #generated_profile = response.choices[0].message.content
        generated_profile = extracted_data
        input_type='Using URL'
        #  تحفظه في db 
        save_data= insert_generated_profile(user_id,None,generated_profile,input_type)
        return {"profile": generated_profile}
    except HTTPException as e:
        raise e  # Forward HTTPException errors (e.g., rate limits)
    




