import streamlit as st
import os
import json
from textblob import TextBlob
from dotenv import load_dotenv
from openai import OpenAI

# ----------------------------------------
# TruthLens – 최신 OpenAI SDK 기반 SNS 허위정보 탐지기
# ----------------------------------------

# 🔐 .env 파일에서 API 키 로드
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 🌐 페이지 설정
st.set_page_config(page_title="TruthLens AI", page_icon="🧠", layout="centered")

st.write("✅ TruthLens 앱이 정상적으로 실행되었습니다.")
st.write("🔑 API Key 로드 상태:", bool(os.getenv("OPENAI_API_KEY")))

st.title("🧠 TruthLens – GPT 기반 SNS 허위정보 탐지기")
st.caption("AI가 SNS 게시물의 신뢰도, 과장 여부, 감정 상태를 분석합니다.")

# ✏️ 사용자 입력
user_post = st.text_area("게시물 내용을 입력하세요:")

# 📁 데이터 폴더 자동 생성
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
LEARNING_FILE = os.path.join(DATA_DIR, "learning_data.json")

# 🧩 기본 감정 분석 + 키워드 탐지
def detect_post(post):
    suspicious_keywords = ['조작', '충격', '폭로', '믿기 힘든', '진짜일까', '음모']
    found = [word for word in suspicious_keywords if word in post]
    sentiment = TextBlob(post).sentiment.polarity
    return {"suspicious": bool(found), "keywords": found, "sentiment": sentiment}

# 🤖 GPT-4 Turbo 기반 신뢰도 분석 (최신 SDK 방식)
def interpret_post_with_gpt(post):
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "너는 SNS 게시물의 신뢰도를 평가하는 AI야. 허위·과장·중립 중 하나로 요약해서 판단해줘."},
                {"role": "user", "content": post}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"⚠️ GPT 분석 오류 발생: {e}")
        return "GPT 오류"

# 💾 학습 데이터 저장
def save_learning_data(post, gpt_label, detection_result):
    record = {
        "post": post,
        "GPT_label": gpt_label,
        "suspicious": detection_result["suspicious"],
        "keywords": detection_result["keywords"],
        "sentiment": detection_result["sentiment"]
    }

    try:
        with open(LEARNING_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = []

    data.append(record)

    with open(LEARNING_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# 🚀 실행 버튼
if st.button("분석하기"):
    if not user_post.strip():
        st.warning("게시물을 입력해주세요.")
    else:
        detection_result = detect_post(user_post)
        gpt_label = interpret_post_with_gpt(user_post)
        save_learning_data(user_post, gpt_label, detection_result)

        st.subheader("🔍 분석 결과")
        st.write(f"**탐지된 키워드:** {', '.join(detection_result['keywords']) if detection_result['keywords'] else '없음'}")
        st.write(f"**감정 점수:** {round(detection_result['sentiment'], 2)}")
        st.write(f"**GPT 판단 결과:** {gpt_label}")

        if any(word in gpt_label for word in ["허위", "과장", "조작", "위험", "거짓"]):
            st.error("⚠️ 경고: 허위 또는 과장 가능성이 있습니다.")
        else:
            st.success("✅ 신뢰할 수 있는 게시물로 판단됩니다.")
