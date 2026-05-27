import json
from pathlib import Path

import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS

from langchain_huggingface import HuggingFaceEmbeddings

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
# API KEY
# =========================================================

try:

    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

except Exception:

    st.error(
        """
        GOOGLE_API_KEY not found.

        Go to:
        App Settings → Secrets

        Add:
        GOOGLE_API_KEY="your_api_key"
        """
    )

    st.stop()

# =========================================================
# FILE PATH
# =========================================================

BASE_DIR = Path(__file__).parent

COURSE_BROCHURE = (
    BASE_DIR / "prompts" / "coursebrochure.pdf"
)

# =========================================================
# GEMINI MODEL
# =========================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.8,
    google_api_key=GOOGLE_API_KEY
)

# =========================================================
# DIPLOMAS
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

# =========================================================
# CATEGORIES
# =========================================================

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

# =========================================================
# SOLUTION TYPES
# =========================================================

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

Retrieved Diploma Knowledge:
{retrieved_context}

Student Inputs:
Diploma: {diploma}
Sustainability Category: {category}
Sustainability Concern: {concern}
Preferred Solution Type: {solution}

Generate THREE sustainability project ideas.

Requirements:
- strongly align with the diploma
- realistic and achievable within 3 months
- solve the sustainability concern
- follow the selected solution type strictly
- practical for diploma students
- concise but meaningful
- avoid generic ideas

Solution Rules:

If the solution type is "Digital Prototype":
- generate apps, AI systems, websites, dashboards, smart platforms, automation systems, or IoT interfaces

If the solution type is "Physical Prototype":
- generate physical products, smart devices, engineering systems, machines, wearables, or environmental hardware

If the solution type is "Social Campaign":
- focus on awareness, behavioural change, outreach, education, storytelling, participation, or engagement

Output ONLY valid JSON.

Format:
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
# CSS
# =========================================================

st.markdown(
    """
    <style>

    .title {
        text-align:center;
        font-size:52px;
        font-weight:700;
    }

    .subtitle {
        text-align:center;
        font-size:24px;
        color:#6B7280;
    }

    .section-title {
        font-size:48px;
        font-weight:700;
    }

    .section-desc {
        font-size:22px;
        color:#6B7280;
    }

    .idea-card {
        background-color:white;
        border-radius:20px;
        padding:40px;
        border:1px solid #E5E7EB;
    }

    .idea-title {
        font-size:32px;
        font-weight:700;
        text-align:center;
        margin-bottom:25px;
    }

    .idea-text {
        font-size:22px;
        line-height:1.8;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# VECTOR DATABASE
# =========================================================

@st.cache_resource
def load_vectorstore():

    loader = PyMuPDFLoader(
        str(COURSE_BROCHURE)
    )

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    split_docs = splitter.split_documents(
        documents
    )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        split_docs,
        embeddings
    )

    return vectorstore

# =========================================================
# RETRIEVE CONTEXT
# =========================================================

def retrieve_context(diploma):

    vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    docs = retriever.invoke(diploma)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    return context

# =========================================================
# GENERATE IDEAS
# =========================================================

def generate_ideas():

    retrieved_context = retrieve_context(
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

            content = content.replace(
                "```json",
                ""
            )

            content = content.replace(
                "```",
                ""
            )

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

    st.title("Hi! I'm SCAle.")

    st.write(
        "I will help you explore sustainability project ideas tailored to your diploma and interests."
    )

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

    st.header("What is your diploma?")

    diploma = st.selectbox(
        "Select your diploma",
        DIPLOMAS
    )

    if st.button("Continue →"):

        st.session_state.diploma = diploma

        st.session_state.page = "category"

        st.rerun()

# =========================================================
# CATEGORY PAGE
# =========================================================

elif st.session_state.page == "category":

    st.header(
        "What sustainability category interests you?"
    )

    category = st.selectbox(
        "Select sustainability category",
        CATEGORIES
    )

    if st.button("Continue →"):

        st.session_state.category = category

        st.session_state.page = "concern"

        st.rerun()

# =========================================================
# CONCERN PAGE
# =========================================================

elif st.session_state.page == "concern":

    st.header(
        "What sustainability problem would you like to solve?"
    )

    concern = st.text_area(
        "Sustainability concern",
        height=220,
        max_chars=200
    )

    if st.button("Continue →"):

        if concern.strip() == "":

            st.warning(
                "Please enter a sustainability concern."
            )

        else:

            st.session_state.concern = concern

            st.session_state.page = "solution"

            st.rerun()

# =========================================================
# SOLUTION PAGE
# =========================================================

elif st.session_state.page == "solution":

    st.header(
        "Which solution format are you interested in developing?"
    )

    solution = st.selectbox(
        "Select Solution Type",
        SOLUTION_TYPES
    )

    if st.button("Submit"):

        st.session_state.solution = solution

        with st.spinner(
            "Generating project ideas..."
        ):

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

    st.title("Project Ideas!")

    st.subheader(
        ideas[current]["title"]
    )

    st.write(
        ideas[current]["idea"]
    )

    col1, col2 = st.columns(2)

    with col1:

        if st.button("⬅ Previous"):

            if current > 0:

                st.session_state.current_idea -= 1

                st.rerun()

    with col2:

        if st.button("Next ➡"):

            if current < len(ideas) - 1:

                st.session_state.current_idea += 1

                st.rerun()

    st.write(
        f"{current + 1} / {len(ideas)}"
    )

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
