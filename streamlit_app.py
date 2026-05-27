import json
import os
import re
from pathlib import Path

import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="SCAle",
    page_icon="🌱",
    layout="wide"
)

# =========================================================
# FILE PATHS
# =========================================================

BASE_DIR = Path(__file__).parent

PROMPTS_DIR = BASE_DIR / "prompts"

COURSE_BROCHURE = PROMPTS_DIR / "coursebrochure.pdf"

# =========================================================
# GEMINI SETUP
# =========================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found.")
    st.stop()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.8,
    google_api_key=GOOGLE_API_KEY
)

# =========================================================
# DATA
# =========================================================

DIPLOMAS = [
    "Diploma in Chemical Engineering",
    "Diploma in Food, Nutrition & Culinary Science",
    "Diploma in Medical Biotechnology",
    "Diploma in Pharmaceutical Science",
    "Diploma in Veterinary Technology",
    "Diploma in Communication Design",
    "Diploma in Digital Film & Television",
    "Diploma in Interior Architecture & Design",
    "Diploma in Fashion Management & Design",
    "Diploma in Product Experience & Design",
    "Diploma in Aerospace Electronics",
    "Diploma in Aerospace Engineering",
    "Diploma in Aviation Management",
    "Diploma in Computer Engineering",
    "Diploma in Architectural Technology and Building Services",
    "Diploma in Electrical and Electronics Engineering",
    "Diploma in Business Process and System Engineering",
    "Diploma in Integrated Facility Management",
    "Diploma in Mechatronics",
    "Diploma in Big Data & Analytics",
    "Diploma in Cybersecurity & Digital Forensics",
    "Diploma in Information Technology",
    "Diploma in Applied Artificial Intelligence",
    "Diploma in Immersive Media and Game Development",
    "Diploma in Accountancy & Finance",
    "Diploma in Business",
    "Diploma in Communications & Media Management",
    "Diploma in Culinary Arts & Management",
    "Diploma in Hospitality & Tourism Management",
    "Diploma in International Trade & Logistics",
    "Diploma in Law & Management",
    "Diploma in Marketing",
    "Diploma in Early Childhood Development & Education",
    "Diploma in Psychology Studies",
    "Diploma in Social Science in Gerontology"
]

CATEGORIES = [
    "Circular Economy",
    "Liveable City and Community",
    "Green Buildings",
    "Renewable Energy",
    "Green Finance and Impact Investing",
    "Sustainable Food Systems",
    "Sustainable Materials and Packaging",
    "Green Transportation",
    "Sustainable Tourism",
    "Green Economy Opportunities",
    "Waste Management and Recycling",
    "Biodiversity and Conservation"
]

SOLUTION_TYPES = [
    "Digital Prototype",
    "Physical Prototype",
    "Social Campaign"
]

# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = r'''
You are an AI sustainability learning assistant for diploma students in Singapore.

Your role is to generate sustainability project ideas that strongly reflect the student's diploma expertise while solving a real sustainability concern.

Retrieved Diploma Knowledge:
{retrieved_context}

Student Inputs:
Diploma: {diploma}
Sustainability Category: {category}
Sustainability Concern: {concern}
Preferred Solution Type: {solution}

Core Objective:
Generate a sustainability solution that:
- solves the sustainability concern realistically
- strongly applies diploma-related knowledge and skills
- remains practical and useful for ordinary users
- can be completed within 3 months by diploma students
- shows contribution opportunities for a team of 5 students
- feels innovative and implementable

STRICT CONSISTENCY RULE:
The diploma specialization and preferred solution type MUST remain fully consistent throughout the generated idea.

Solution Type Enforcement Rules:

If Preferred Solution Type is "Digital Prototype":
- generate software-based systems such as apps, AI systems, dashboards, websites, automation systems, or smart digital platforms

If Preferred Solution Type is "Physical Prototype":
- generate tangible physical products, smart devices, engineering systems, machines, wearables, smart bins, filtration devices, packaging innovations, or environmental hardware

If Preferred Solution Type is "Social Campaign":
- generate awareness-driven solutions focused on behavioural change, community engagement, storytelling, sustainability education, participation systems, or outreach initiatives

STRICT RESTRICTIONS:
- Do NOT generate digital-only solutions for Physical Prototype
- Do NOT generate hardware-focused solutions for Digital Prototype
- Do NOT generate apps or devices for Social Campaign unless they are only supporting tools
- Avoid overly futuristic ideas
- Avoid unrealistic solutions

Output Rules:
- Generate THREE distinct sustainability project ideas
- Keep each idea concise
- Return ONLY valid JSON

Return Format:
[
  {{
    "title": "Idea 1",
    "idea": "..."
  }},
  {{
    "title": "Idea 2",
    "idea": "..."
  }},
  {{
    "title": "Idea 3",
    "idea": "..."
  }}
]
'''

# =========================================================
# SESSION STATE
# =========================================================

if "page" not in st.session_state:
    st.session_state.page = "welcome"

if "diploma" not in st.session_state:
    st.session_state.diploma = ""

if "category" not in st.session_state:
    st.session_state.category = ""

if "concern" not in st.session_state:
    st.session_state.concern = ""

if "solution" not in st.session_state:
    st.session_state.solution = ""

if "ideas" not in st.session_state:
    st.session_state.ideas = []

if "current_idea" not in st.session_state:
    st.session_state.current_idea = 0

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #F6F6F6;
    }

    .title {
        text-align: center;
        font-size: 48px;
        font-weight: 700;
        color: #1F1F1F;
        margin-top: 40px;
    }

    .subtitle {
        text-align: center;
        font-size: 24px;
        color: #6B7280;
        margin-bottom: 40px;
    }

    .section-title {
        font-size: 48px;
        font-weight: 700;
        color: #1F1F1F;
        margin-bottom: 10px;
    }

    .section-desc {
        font-size: 22px;
        color: #6B7280;
        margin-bottom: 30px;
    }

    .idea-card {
        background-color: white;
        padding: 40px;
        border-radius: 20px;
        border: 1px solid #E5E7EB;
        min-height: 350px;
    }

    .idea-title {
        text-align: center;
        font-size: 34px;
        font-weight: 700;
        margin-bottom: 25px;
    }

    .idea-text {
        font-size: 22px;
        line-height: 1.8;
        color: #333333;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# LOAD RAG DATABASE
# =========================================================

@st.cache_resource
def load_vectorstore():

    loader = PyMuPDFLoader(str(COURSE_BROCHURE))

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        docs,
        embeddings
    )

    return vectorstore

# =========================================================
# RETRIEVE DIPLOMA CONTEXT
# =========================================================

def retrieve_diploma_context(diploma):

    vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    docs = retriever.invoke(diploma)

    context = "\n\n".join([doc.page_content for doc in docs])

    return context

# =========================================================
# GENERATE IDEAS
# =========================================================

def generate_ideas():

    retrieved_context = retrieve_diploma_context(
        st.session_state.diploma
    )

    prompt = SYSTEM_PROMPT.format(
        retrieved_context=retrieved_context,
        diploma=st.session_state.diploma,
        category=st.session_state.category,
        concern=st.session_state.concern,
        solution=st.session_state.solution
    )

    response = llm.invoke(prompt)

    try:

        content = response.content.strip()

        if content.startswith("```json"):
            content = content.replace("```json", "")
            content = content.replace("```", "")

        ideas = json.loads(content)

        return ideas

    except Exception:

        return [
            {
                "title": "Error",
                "idea": response.content
            }
        ]

# =========================================================
# WELCOME PAGE
# =========================================================

if st.session_state.page == "welcome":

    st.markdown("<div style='height:60px'></div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:

        st.markdown(
            "<div class='title'>Hi! I'm SCAle.</div>",
            unsafe_allow_html=True
        )

        st.markdown(
            "<div class='subtitle'>I will help you to explore sustainability project ideas tailored to your diploma and interests. Let's get started.</div>",
            unsafe_allow_html=True
        )

        st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)

        if st.button(
            "Start Your Project Ideas",
            use_container_width=True
        ):
            st.session_state.page = "diploma"
            st.rerun()

# =========================================================
# DIPLOMA PAGE
# =========================================================

elif st.session_state.page == "diploma":

    if st.button("←"):
        st.session_state.page = "welcome"
        st.rerun()

    st.markdown("<div style='height:70px'></div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='section-title'>What is your diploma?</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='section-desc'>This helps me to tailor sustainability project ideas to your field of study.</div>",
        unsafe_allow_html=True
    )

    diploma = st.selectbox(
        "Select your diploma",
        DIPLOMAS
    )

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    if st.button("Continue →"):

        st.session_state.diploma = diploma
        st.session_state.page = "category"

        st.rerun()

# =========================================================
# CATEGORY PAGE
# =========================================================

elif st.session_state.page == "category":

    if st.button("←"):
        st.session_state.page = "diploma"
        st.rerun()

    st.markdown("<div style='height:70px'></div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='section-title'>What sustainability category interests you?</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='section-desc'>This allows sustainability project ideas align to your focus areas.</div>",
        unsafe_allow_html=True
    )

    category = st.selectbox(
        "Select sustainability category",
        CATEGORIES
    )

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    if st.button("Continue →"):

        st.session_state.category = category
        st.session_state.page = "concern"

        st.rerun()

# =========================================================
# CONCERN PAGE
# =========================================================

elif st.session_state.page == "concern":

    if st.button("←"):
        st.session_state.page = "category"
        st.rerun()

    st.markdown("<div style='height:70px'></div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='section-title'>What sustainability problem would you like to solve?</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='section-desc'>Share a problem or challenge you have noticed in school, community, or daily life.</div>",
        unsafe_allow_html=True
    )

    concern = st.text_area(
        "Sustainability concern",
        max_chars=200,
        height=220
    )

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    if st.button("Continue →"):

        if concern.strip() == "":
            st.warning("Please enter your sustainability concern.")

        else:

            st.session_state.concern = concern
            st.session_state.page = "solution"

            st.rerun()

# =========================================================
# SOLUTION PAGE
# =========================================================

elif st.session_state.page == "solution":

    if st.button("←"):
        st.session_state.page = "concern"
        st.rerun()

    st.markdown("<div style='height:70px'></div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='section-title'>Which solution format are you interested in developing?</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='section-desc'>This helps me to suggest the right type of project for you.</div>",
        unsafe_allow_html=True
    )

    solution = st.selectbox(
        "Select Solution Type",
        SOLUTION_TYPES
    )

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    if st.button("Submit"):

        st.session_state.solution = solution

        with st.spinner("Generating project ideas..."):

            st.session_state.ideas = generate_ideas()

        st.session_state.current_idea = 0
        st.session_state.page = "results"

        st.rerun()

# =========================================================
# RESULTS PAGE
# =========================================================

elif st.session_state.page == "results":

    ideas = st.session_state.ideas

    current = st.session_state.current_idea

    st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)

    st.markdown(
        "<div style='text-align:center;font-size:30px;'>💡</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div style='text-align:center;font-size:28px;font-weight:700;'>Here are your</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div style='text-align:center;font-size:54px;font-weight:700;color:#2F5F38;'>Project Ideas!</div>",
        unsafe_allow_html=True
    )

    st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 8, 1])

    with col1:

        if st.button("⬅"):

            if current > 0:

                st.session_state.current_idea -= 1

                st.rerun()

    with col2:

        st.markdown(
            f"""
            <div class='idea-card'>
                <div class='idea-title'>
                    {ideas[current]['title']}
                </div>

                <div class='idea-text'>
                    {ideas[current]['idea']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"<div style='text-align:center;font-size:20px;margin-top:20px'>{current + 1} / {len(ideas)}</div>",
            unsafe_allow_html=True
        )

    with col3:

        if st.button("➡"):

            if current < len(ideas) - 1:

                st.session_state.current_idea += 1

                st.rerun()

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    if st.button(
        "Start Over",
        use_container_width=True
    ):

        st.session_state.page = "welcome"
        st.session_state.diploma = ""
        st.session_state.category = ""
        st.session_state.concern = ""
        st.session_state.solution = ""
        st.session_state.ideas = []
        st.session_state.current_idea = 0

        st.rerun()
