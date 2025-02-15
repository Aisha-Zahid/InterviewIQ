import streamlit as st
import os
import whisper
from groq import Groq
import google.generativeai as genai
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import tempfile
from io import BytesIO

# Initialize models and APIs
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

# Initialize clients
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Streamlit UI Config
st.set_page_config(page_title="InterviewIQ", page_icon="🎙️", layout="wide")
st.markdown("""<style>.report-card {padding:20px; border-radius:10px; margin:10px 0;}</style>""", unsafe_allow_html=True)

def transcribe_audio(audio_path):
    model = load_whisper_model()
    result = model.transcribe(audio_path)
    return result["text"]

def analyze_with_groq(text):
    prompt = f"""Analyze this interview response:
    {text}
    Provide feedback with:
    - 3 Strengths
    - 3 Improvements
    - Overall Score (1-10)
    """
    response = groq_client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def analyze_with_gemini(text):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(f"Analyze interview response: {text}")
    return response.text

def extract_score(feedback):
    # Extract score from feedback text (assumes format: "Overall Score (1-10): X")
    for line in feedback.split("\n"):
        if "Overall Score" in line:
            try:
                return int(line.split(":")[-1].strip())
            except ValueError:
                return None
    return None

def plot_performance_chart(score_history):
    plt.figure(figsize=(8, 4))
    plt.plot(score_history, marker='o', linestyle='-', color='b', label='Interview Score')
    plt.xlabel('Interview Session')
    plt.ylabel('Score (1-10)')
    plt.title('Performance Over Time')
    plt.ylim(0, 10)
    plt.legend()
    st.pyplot(plt)

def main():
    # st.title("🎙️ InterviewIQ - Local AI Interview Coach")
    # st.markdown("### No OpenAI Required • Full Privacy")
    st.markdown(
    """
    <style>
        .title {
            text-align: center;
            font-size: 2.8em;
            font-weight: bold;
            background: linear-gradient(90deg, #8A2BE2, #DA70D6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle {
            text-align: center;
            font-size: 1.3em;
            color: #555;
            margin-top: -10px;
        }
        .intro {
            text-align: center;
            font-size: 1.1em;
            color: #fff;
            font-style: italic;
            padding: 10px;
            background: rgba(138, 43, 226, 0.2);
            border-radius: 10px;
        }
    </style>

    <h1 class="title">🎙️ InterviewIQ: Your Personal AI Coach</h1>
    <h3 style="text-align: center; color: #555;">
        No WiFi? No Problem!😎 Your AI Interview Coach, Anytime, Anywhere.
    </h3>
    <p class="intro">🪄 Ready to crush your next interview? No awkward silences, no robotic scripts—just pure AI-driven coaching that hypes you up, polishes your skills, and gets you job-ready like a boss 🏆</p>
    
    """,  
    
    unsafe_allow_html=True
)


    if 'history' not in st.session_state:
        st.session_state.history = []
    
    with st.expander("🎤 Start Analysis", expanded=True):
        input_method = st.radio("Input Method:", ("Audio Upload", "Text Input"))
        transcription = ""
        
        if input_method == "Audio Upload":
            audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3"])
            if audio_file:
                # Display audio player
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format=f'audio/{audio_file.type.split("/")[-1]}')
                
                # Write to temporary file for processing
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(audio_bytes)
                    transcription = transcribe_audio(tmp_file.name)
        else:
            transcription = st.text_area("Paste Transcript:", height=150)
    
    if transcription:
        col1, col2 = st.columns([3,1])
        with col1:
            st.markdown("### 📝 Transcription")
            st.write(transcription)
            
        with col2:
            st.markdown("### ⚙️ Settings")
            llm_choice = st.radio("AI Analyst:", ("Groq", "Gemini"))
            
            if st.button("🚀 Analyze"):
                with st.spinner("Analyzing..."):
                    feedback = analyze_with_groq(transcription) if llm_choice == "Groq" else analyze_with_gemini(transcription)
                    score = extract_score(feedback) if feedback else None
                    
                    record = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "transcription": transcription,
                        "feedback": feedback,
                        "score": score
                    }
                    st.session_state.history.insert(0, record)
        
        if 'record' in locals():
            st.markdown("---")
            st.markdown("## 📊 Analysis Results")
            
            col1, col2 = st.columns([2,1])
            with col1:
                st.markdown("### 🔍 Feedback")
                st.markdown(f'<div class="report-card">{feedback}</div>', unsafe_allow_html=True)
            
            if score is not None:
                with col2:
                    st.markdown("### 📈 Performance Chart")
                    scores = [entry['score'] for entry in st.session_state.history if entry['score'] is not None]
                    if scores:
                        plot_performance_chart(scores)
            
            st.markdown("---")
            st.markdown("## 📚 Analysis History")
            for idx, entry in enumerate(st.session_state.history):
                with st.expander(f"Analysis #{idx+1} - {entry['timestamp']}"):
                    col1, col2 = st.columns([2,1])
                    with col1:
                        st.write("**Transcription:**", entry['transcription'])
                        st.write("**Feedback:**", entry['feedback'])
                    if entry['score'] is not None:
                        with col2:
                            st.write(f"**Score:** {entry['score']}")
                            
if __name__ == "__main__":
    main()


