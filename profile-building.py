from fastapi import FastAPI
from database import fetch_profile_data ,insert_generated_profile
import os
from dotenv import load_dotenv
import json
from openai import OpenAI
import uvicorn

# تحميل متغيرات البيئة
load_dotenv()

app = FastAPI()

# إنشاء مجلد للملفات لو مش موجود
os.makedirs("generated_pdf", exist_ok=True)

# إعداد اتصال بقاعدة البيانات
api_key=os.getenv("OPENAI_API_KEY")

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
    data=all_data[-1]

    # نقرأها من جديد للتجربة
    loaded_examples = load_examples_from_json()

    # توليد البروفايل
    generated_profile = generate_profile_model(data, loaded_examples)

    save_data= insert_generated_profile(user_id,data['organization_name'],generated_profile)

    return {"generated_profile":generated_profile}
