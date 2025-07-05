import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Streamlit setup
st.set_page_config(page_title="AI Medical Assistant", layout="centered", page_icon="🩺")
st.title("🩺 AI Medical Image Assessment")
st.caption("Built with ❤️ by Shivam Bhardwaj")

# Sidebar
with st.sidebar:
    st.header("ℹ️ How It Works")
    st.markdown("""
    This app could be used to:
    - Analyze medical images
    - Suggest possible health issues
    - Explain results simply if needed
    """)
    st.success("Data is processed locally and not stored.")

# Prompt for analysis
ANALYSIS_PROMPT = """
You are a senior medical expert. Analyze the uploaded image and provide:
1. Medical findings
2. Potential issues or anomalies
3. Recommended next steps

If image is unclear, say so.
Always include this at the end:  
"**Disclaimer**: Please consult a licensed physician for accurate diagnosis."
"""

# Function: Image analysis using gemini-1.5-flash
def analyze_image(image_pil):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content([ANALYSIS_PROMPT, image_pil])
        return response.text
    except Exception as e:
        return f"⚠️ Error during analysis: {e}"

# Function: ELI5
def explain_eli5(text):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(f"Explain the following in very simple terms like I'm 5 years old:\n{text}")
        return response.text
    except Exception as e:
        return f"⚠️ Error during explanation: {e}"

# File upload
uploaded_file = st.file_uploader("📤 Upload a medical image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🧪 Analyze Image"):
        with st.spinner("Analyzing with Gemini 1.5 Flash..."):
            result = analyze_image(image)
            st.session_state["analysis"] = result
            st.success("✅ Analysis complete!")

# Display results
if "analysis" in st.session_state:
    st.markdown("## 📋 AI Diagnosis")
    st.markdown(st.session_state["analysis"], unsafe_allow_html=True)

    with st.expander("👶 Simplify It - ELI5"):
        if st.button("🧠 Explain like I’m 5"):
            with st.spinner("Simplifying..."):
                simplified = explain_eli5(st.session_state["analysis"])
                st.markdown("### 🧒 Simple Explanation")
                st.markdown(simplified, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("🔧 **Built by Shivam Bhardwaj** | Using Gemini 1.5 Flash")
