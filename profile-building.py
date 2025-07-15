from fastapi import FastAPI
from database import fetch_profile_data ,insert_generated_profile
from pydantic import BaseModel
from typing import Optional
from fastapi.responses import FileResponse
#import mysql.connector
#import datetime
import os
#from weasyprint import HTML
from dotenv import load_dotenv
import json
from openai import OpenAI

# تحميل متغيرات البيئة
load_dotenv()

app = FastAPI()

# إنشاء مجلد للملفات لو مش موجود
os.makedirs("generated_pdf", exist_ok=True)

# إعداد اتصال بقاعدة البيانات
api_key=os.getenv("OPENAI_API_KEY")

#host = os.getenv("DB_HOST")
#port = os.getenv("DB_PORT")


def load_examples_from_json(json_path="example_profiles.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_profile_model(data, examples):

    client = OpenAI(api_key=api_key)
    examples_text = "\n\n".join(examples[:2])  # نرسل أول مثالين فقط لتقليل الطول
    prompt=f'''أنت خبير متخصص في كتابة الملفات التعريفية للشركات (Company Profiles)، وتعمل كمستشار استراتيجي في تطوير الهوية المؤسسية والعرض الاحترافي للخدمات.
ستتلقى مجموعة من الملفات تتضمن محتوى خام وتعريفي من عدة شركات{examples_text}، دورك هو أن تحلل هذه الملفات بدقة، وتستخلص منها الأسلوب الاحترافي الأمثل لبناء ملف تعريفي متميز ومتكامل لشركة معلوماتها فى {data}مع ذكر الرؤيه والرساله فى فقرات منفصله .
المطلوب:
    كتابة ملف تعريفي احترافي للشركة بأسلوب عصري وجذاب، يُراعي اللغة المؤسسية، ويُبرز الهوية والمكانة التنافسية.
    لا تعتمد على هيكل جاهز، بل ابتكر ترتيبًا منطقيًا وتدريجيًا للمحتوى يُناسب الشركة ومجالها.
    اجعل الملف غنيًا بالتفاصيل، ومتوازنًا بين النصوص التسويقية، والمحتوى المعلوماتي، والمزايا التنافسية.
    ضمّن أقسامًا مثل: من نحن، ما الذي نُقدمه، لماذا نحن، أعمالنا، خدماتنا، أسلوبنا، وغيرها إن وجدت مناسبة.
    اجعل الكتابة سلسة، متماسكة، ومتناسقة بصريًا ومضمونيًا.
    افترض أنك تكتب الملف ليُستخدم في طباعة فاخرة، وعرض إلكتروني، وعرض تقديمي.
سيُستخدم الملف لاحقًا من قِبل جهات استثمارية وعملاء محتملين، لذا يجب أن يُجسّد هوية الشركة وقوتها.
'''

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content


@app.get("/profile-generating-tool/{user_id}")
def profile_generating_tool(user_id):

    all_data= fetch_profile_data(user_id)
    print("allllllllllllllllll data",all_data,len(all_data))
    data=all_data[-1]

    # نقرأها من جديد للتجربة
    loaded_examples = load_examples_from_json()

    # توليد البروفايل
    generated_profile = generate_profile_model(data, loaded_examples)
    print("\n✅ النص الناتج:")
    print(generated_profile)

    save_data= insert_generated_profile(user_id,data['organization_name'],generated_profile)
    print(save_data)
    return generated_profile





'''

class ProfileInput(BaseModel):
    user_id: int
    company_name: Optional[str] = None
    year: Optional[str] = None
    location: Optional[str] = None
    vision: Optional[str] = None
    message: Optional[str] = None
    achievements: Optional[str] = None

@app.post("/fetch-from-url")
async def fetch_from_url(url: str):
    # مثال ثابت بدلاً من scraping
    return {
        "company_name": "شركة افتراضية",
        "year": "2022",
        "location": "الرياض",
        "vision": "أن نكون الأفضل",
        "message": "خدمة عملائنا بكل احترافية"
    }

@app.post("/generate-profile")
async def generate_profile(data: ProfileInput):
    now = datetime.datetime.now().isoformat()
    conn = get_connection()
    cursor = conn.cursor()

    # حفظ البيانات في DB
    cursor.execute("""
    INSERT INTO profiles 
    (user_id, company_name, year, location, vision, message, achievements, created_at)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        data.user_id, data.company_name, data.year, data.location, 
        data.vision, data.message, data.achievements, now
    ))
    conn.commit()
    profile_id = cursor.lastrowid

    # توليد HTML باستخدام القالب
    from jinja2 import Environment, FileSystemLoader
    env = Environment(loader=FileSystemLoader('templates'))
    template = env.get_template('profile_template.html')
    html_content = template.render(
        company_name=data.company_name,
        year=data.year,
        location=data.location,
        vision=data.vision,
        message=data.message,
        achievements=data.achievements
    )

    # توليد PDF
    pdf_path = f"generated/profile_{profile_id}.pdf"
    HTML(string=html_content).write_pdf(pdf_path)

    # تحديث DB
    cursor.execute("""
    UPDATE profiles SET generated_html=%s, pdf_path=%s WHERE id=%s
    """, (html_content, pdf_path, profile_id))
    conn.commit()
    cursor.close()
    conn.close()

    return {"profile_id": profile_id}

@app.get("/download-profile/{profile_id}")
async def download_profile(profile_id: int):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT pdf_path FROM profiles WHERE id=%s", (profile_id,))
    row = cursor.fetchone()
    cursor.close()
    conn.close()

    if row and os.path.exists(row[0]):
        return FileResponse(path=row[0], filename=f"profile_{profile_id}.pdf", media_type='application/pdf')
    return {"error": "Not found"}
    
    '''
