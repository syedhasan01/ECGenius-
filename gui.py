import streamlit as st
import pandas as pd
import joblib
import base64
import requests
import time
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ CSS Styling -------------------
def add_custom_css():
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;500;700&display=swap');
        html, body, [class*="css"] {{
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(to right, #fceabb, #f8b500);
        }}
        .stApp {{
            padding: 2rem;
        }}
        h1, h2, h3, h4 {{
            color: #0d47a1;
        }}
        .stButton > button {{
            background-color: #ff6f00;
            color: white;
            padding: 0.6em 1.2em;
            border-radius: 8px;
            font-weight: 600;
        }}
        .highlight-box {{
            background-color: #ffffffcc;
            padding: 1.5em;
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            margin-bottom: 1.5em;
        }}
        .chatbot-box {{
            background-color: #fff9e6;
            padding: 1.2em;
            border-radius: 12px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        }}
        </style>
    """, unsafe_allow_html=True)

# ------------------ Load Model -------------------
@st.cache_resource
def load_model():
    return joblib.load("ecg_random_forest_model.pkl")

# ------------------ Hospital Recommendations -------------------
def get_hospitals_by_zone(zone):
    return {
        "North": ["🏥 Abbasi Shaheed Hospital", "🏥 Mamji Hospital", "🏥 Karachi Institute of Heart Diseases (KIHD)"],
        "South": ["🏥 Aga Khan University Hospital", "🏥 South City Hospital", "🏥 Indus Hospital"],
        "East": ["🏥 Liaquat National Hospital", "🏥 Patel Hospital", "🏥 Hill Park General Hospital"],
        "West": ["🏥 Civil Hospital", "🏥 Qatar Hospital", "🏥 Sindh Government Hospital, New Karachi"],
        "Central": ["🏥 Tabba Heart Institute", "🏥 Ziauddin Hospital", "🏥 Saifee Hospital"]
    }.get(zone, [])

# ------------------ Chatbot Logic -------------------
faq_data = {
    "what is ecg": "ECG (Electrocardiogram) is a test that records the electrical signals of your heart.",
    "what is heart disease": "Heart disease refers to various conditions that affect the heart's structure and function.",
    "how to prevent heart disease": "Maintain a healthy lifestyle, regular exercise, and avoid smoking. Also, control blood pressure and cholesterol levels.",
    "what are symptoms of heart disease": "Common symptoms include chest pain, shortness of breath, fatigue, and irregular heartbeat.",
    "what does abnormal mean": "An abnormal ECG result may indicate irregularities in the heart's rhythm or structure."
}

@st.cache_resource
def get_vectorizer_and_matrix():
    vectorizer = TfidfVectorizer()
    questions = list(faq_data.keys())
    tfidf_matrix = vectorizer.fit_transform(questions)
    return vectorizer, tfidf_matrix, questions

def ecg_chatbot_response(user_input):
    vectorizer, tfidf_matrix, questions = get_vectorizer_and_matrix()
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, tfidf_matrix).flatten()
    index = similarity.argmax()
    if similarity[index] > 0.3:
        return faq_data[questions[index]]
    return "Sorry, I couldn't understand your question. Please try asking something related to ECG or heart disease."

# ------------------ Fetch from Ubidots -------------------
def fetch_from_ubidots():
    UBIDOTS_TOKEN = "BBUS-1II7XFOZFQtBxtbEikaRK15z4wdyVd"
    DEVICE_LABEL = "esp32"
    VARIABLE_LABEL = "sensor"
    PATIENT_ID = "P001"

    headers = {"X-Auth-Token": UBIDOTS_TOKEN}
    end_time = int(time.time() * 1000)
    start_time = end_time - (30 * 24 * 60 * 60 * 1000)

    url = f"https://industrial.api.ubidots.com/api/v1.6/devices/{DEVICE_LABEL}/{VARIABLE_LABEL}/values"
    params = {
        "start": start_time,
        "end": end_time,
        "page_size": 450
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        results = response.json().get("results", [])
        sensor_values = [entry["value"] for entry in results]

        if len(sensor_values) >= 450:
            sensor_values = sensor_values[:450]

            features = []
            for i in range(18):
                window_values = sensor_values[i * 25:(i + 1) * 25]
                features.append({
                    "PatientID": PATIENT_ID,
                    "Window": i + 1,
                    "Min": min(window_values),
                    "Max": max(window_values),
                    "Mean": sum(window_values) / len(window_values),
                    "Std": pd.Series(window_values).std()
                })

            df_features = pd.DataFrame(features)
            df_features.to_excel("ecg_features_18_windows.xlsx", index=False)
            st.success("✅ Data fetched and saved as ecg_features_18_windows.xlsx")

            with open("ecg_features_18_windows.xlsx", "rb") as f:
                st.download_button(
                    label="⬇️ Download Fetched ECG File",
                    data=f,
                    file_name="ecg_features_18_windows.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:
            st.error(f"Not enough data: only {len(sensor_values)} values received.")
    else:
        st.error(f"Failed to retrieve data: {response.status_code} - {response.text}")

# ------------------ Main ECG App -------------------
def main_app():
    add_custom_css()
    st.markdown("<h1 style='text-align: center;'>💓 ECGenius</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Upload ECG features to check heart health status.</p>", unsafe_allow_html=True)
    st.markdown("---")

    if st.button("🔄 Fetch ECG Data from Ubidots"):
        fetch_from_ubidots()

    model = load_model()
    uploaded_file = st.file_uploader("📁 Upload Excel file (columns: Min, Max, Mean, Std)", type=["xlsx"])

    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            if {'Min', 'Max', 'Mean', 'Std'}.issubset(df.columns):
                X = df[['Min', 'Max', 'Mean', 'Std']]
                predictions = model.predict(X)
                abnormal_count = (predictions == 1).sum()

                st.subheader("🩺 Prediction Results")
                for idx, pred in enumerate(predictions, 1):
                    label = "Abnormal (1) - Heart Patient" if pred == 1 else "Normal (0) - Healthy"
                    color = "red" if pred == 1 else "green"
                    st.markdown(f"<span style='color:{color}; font-weight:600;'>Window {idx}: {label}</span>", unsafe_allow_html=True)

                st.markdown("---")
                st.subheader("🔍 Summary")
                if abnormal_count > 0:
                    st.error("⚠️ Abnormality detected! Patient may be at risk.")
                    name = st.text_input("Patient Name", placeholder="Enter name")
                    age = st.number_input("Age", min_value=1, max_value=120)
                    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                    zone = st.selectbox("Zone in Karachi", ["North", "South", "East", "West", "Central"])

                    if st.button("🔍 Get Recommendation"):
                        if not name:
                            st.warning("Please enter the patient's name.")
                        else:
                            st.markdown(f"<h3>💔 Diagnosis Report for {name}</h3>", unsafe_allow_html=True)
                            st.write(f"*Age:* {age} | *Gender:* {gender} | *Zone:* {zone}")

                            st.markdown("#### 💊 Suggested Medicines")
                            for med in ["💊 Aspirin", "💊 Atorvastatin", "💊 Metoprolol", "💊 Lisinopril"]:
                                st.markdown(f"- {med}")

                            st.markdown(f"#### 🏥 Hospitals in {zone}")
                            for hospital in get_hospitals_by_zone(zone):
                                st.markdown(f"- {hospital}")

                            st.info("📞 Please consult a certified cardiologist immediately.")
                else:
                    st.success("✅ No abnormalities found. Patient seems healthy.")
                    st.balloons()
            else:
                st.error("Excel must contain 'Min', 'Max', 'Mean', and 'Std' columns.")
        except Exception as e:
            st.error(f"Error reading file: {e}")

    st.markdown("---")
    st.subheader("🧠 ECG Chatbot")
    user_question = st.text_input("Ask a question about ECG or heart disease")
    if st.button("Ask"):
        if user_question.strip():
            response = ecg_chatbot_response(user_question)
            st.markdown(f"*Chatbot:* {response}")
        else:
            st.warning("Please type a question before asking.")

# ------------------ Login Page -------------------
def login_page():
    st.title("🔐 Login to ECGenius")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin123" and password == "admin123":
            st.session_state.authenticated = True
            st.success("✅ Login successful!")
            st.rerun()
        else:
            st.error("❌ Invalid username or password.")

# ------------------ Main Logic -------------------
def main():
    st.set_page_config(page_title="ECGenius", page_icon="💓", layout="centered")
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        main_app()
    else:
        login_page()

if __name__ == "__main__":
    main()