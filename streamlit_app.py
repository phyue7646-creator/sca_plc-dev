import json
import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI

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

        Add this in Streamlit Secrets:

        GOOGLE_API_KEY="your_api_key"
        """
    )

    st.stop()

# =========================================================
# GEMINI MODEL
# =========================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
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
# PROMPT
# =========================================================

SYSTEM_PROMPT = """
You are an AI sustainability learning assistant for diploma students in Singapore.

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
"""

# =========================================================
# SESSION STATE
# =========================================================

if "page" not in st.session_state:
    st.session_state.page = "welcome"

if "ideas" not in st.session_state:
    st.session_state.ideas = []

if "current_idea" not in st.session_state:
    st.session_state.current_idea = 0

# =========================================================
# GENERATE IDEAS
# =========================================================

def generate_ideas(
    diploma,
    category,
    concern,
    solution
):

    prompt = SYSTEM_PROMPT.format(
        diploma=diploma,
        category=category,
        concern=concern,
        solution=solution
    )

    response = llm.invoke(prompt)

    try:

        content = response.content.strip()

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
                "title": "Generation Error",
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
        "Start Your Project Ideas"
    ):

        st.session_state.page = "form"

        st.rerun()

# =========================================================
# FORM PAGE
# =========================================================

elif st.session_state.page == "form":

    diploma = st.selectbox(
        "What is your diploma?",
        DIPLOMAS
    )

    category = st.selectbox(
        "What sustainability category interests you?",
        CATEGORIES
    )

    concern = st.text_area(
        "What sustainability problem would you like to solve?",
        max_chars=200,
        height=200
    )

    solution = st.selectbox(
        "Which solution format are you interested in developing?",
        SOLUTION_TYPES
    )

    if st.button("Generate Ideas"):

        with st.spinner(
            "Generating project ideas..."
        ):

            st.session_state.ideas = generate_ideas(
                diploma,
                category,
                concern,
                solution
            )

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

    if st.button("Start Over"):

        st.session_state.page = "welcome"

        st.session_state.ideas = []

        st.session_state.current_idea = 0

        st.rerun()
