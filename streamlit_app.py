import streamlit as st
import google.generativeai as genai

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="SCAle",
    page_icon="🌱",
    layout="wide"
)

# =========================================================
# GOOGLE GEMINI API
# =========================================================

GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel("gemini-1.5-flash")

# =========================================================
# SESSION STATE
# =========================================================

if "page" not in st.session_state:
    st.session_state.page = "welcome"

if "solution_type" not in st.session_state:
    st.session_state.solution_type = ""

if "concern" not in st.session_state:
    st.session_state.concern = ""

if "category" not in st.session_state:
    st.session_state.category = ""

if "diploma" not in st.session_state:
    st.session_state.diploma = ""

if "ideas" not in st.session_state:
    st.session_state.ideas = []

if "idea_index" not in st.session_state:
    st.session_state.idea_index = 0

# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

.stApp {
    background-color: #F4F4F4;
}

/* LOGO */

.logo-container {
    position: absolute;
    top: 25px;
    left: 45px;
    font-size: 42px;
    font-weight: 700;
    z-index: 999;
}

.logo-dark {
    color: #2F5D3A;
}

.logo-light {
    color: #A8C256;
}

/* MAIN TITLE */

.main-title {
    font-size: 70px;
    font-weight: 700;
    text-align: center;
    color: #1F1F1F;
    margin-top: 80px;
    line-height: 1.1;
}

/* SUBTITLE */

.sub-title {
    font-size: 24px;
    text-align: center;
    color: #666666;
    margin-top: 20px;
    margin-bottom: 40px;
    line-height: 1.6;
}

/* QUESTION PAGE */

.question-title {
    font-size: 58px;
    font-weight: 700;
    color: #1E1E1E;
    margin-top: 80px;
    line-height: 1.2;
}

.question-subtitle {
    font-size: 24px;
    color: #707070;
    margin-top: 18px;
    margin-bottom: 45px;
    line-height: 1.5;
}

/* RESULT */

.result-title {
    text-align: center;
    font-size: 32px;
    font-weight: 600;
    margin-top: 15px;
    color: #1E1E1E;
}

.result-heading {
    text-align: center;
    font-size: 72px;
    font-weight: 700;
    color: #2F5D3A;
    margin-bottom: 20px;
}

/* IDEA BOX */

.idea-box {
    background-color: white;
    border: 2px solid #DDDDDD;
    border-radius: 22px;
    padding: 45px;
    min-height: 420px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.04);
}

.idea-content {
    font-size: 25px;
    line-height: 1.9;
    color: #333333;
}

/* BUTTON */

.stButton>button {
    background-color: #437A4A;
    color: white;
    border-radius: 16px;
    border: none;
    font-size: 22px;
    font-weight: 700;
    padding: 15px 30px;
    transition: 0.3s;
}

.stButton>button:hover {
    background-color: #35633B;
    color: white;
}

/* SELECTBOX */

.stSelectbox div[data-baseweb="select"] {
    border-radius: 10px;
}

/* TEXT AREA */

textarea {
    border-radius: 14px !important;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOGO
# =========================================================

st.markdown(
    """
    <div class='logo-container'>
        <span class='logo-dark'>SCA</span><span class='logo-light'>le</span>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# DATA
# =========================================================

solution_types = [
    "Digital Prototype",
    "Physical Prototype",
    "Social Campaign"
]

categories = [
    "Circular Economy",
    "Renewable Energy",
    "Waste Management and Recycling",
    "Green Transportation",
    "Sustainable Tourism",
    "Biodiversity and Conservation",
    "Green Buildings",
    "Sustainable Food Systems"
]

diplomas = [
    "Diploma in Information Technology",
    "Diploma in Big Data & Analytics",
    "Diploma in Accountancy & Finance",
    "Diploma in Applied AI",
    "Diploma in Chemical Engineering",
    "Diploma in Business",
    "Diploma in Marketing",
    "Diploma in Cybersecurity & Digital Forensics",
    "Diploma in Pharmaceutical Science",
    "Diploma in Communication Design"
]

# =========================================================
# WELCOME PAGE
# =========================================================

if st.session_state.page == "welcome":

    st.markdown(
        """
        <div class='main-title'>
        Hi! I'm 
        <span style='color:#2F5D3A;'>SCA</span><span style='color:#A8C256;'>le</span>.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class='sub-title'>
        I will help you explore sustainability project ideas tailored to your diploma and interests.
        Let's get started.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.image(
        "https://cdn-icons-png.flaticon.com/512/4712/4712109.png",
        width=320
    )

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        if st.button("Start Your Project Ideas"):
            st.session_state.page = "solution"
            st.rerun()

# =========================================================
# SOLUTION PAGE
# =========================================================

elif st.session_state.page == "solution":

    st.markdown(
        "<div class='question-title'>Which solution format are you interested in developing?</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='question-subtitle'>This helps me suggest the right type of project for you.</div>",
        unsafe_allow_html=True
    )

    solution = st.selectbox(
        "Select Solution Type",
        solution_types
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Continue →"):
        st.session_state.solution_type = solution
        st.session_state.page = "concern"
        st.rerun()

# =========================================================
# CONCERN PAGE
# =========================================================

elif st.session_state.page == "concern":

    st.markdown(
        "<div class='question-title'>What sustainability problem would you like to solve?</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='question-subtitle'>Share a problem or challenge you have noticed in school, community, or daily life.</div>",
        unsafe_allow_html=True
    )

    concern = st.text_area(
        "Sustainability Concern",
        max_chars=200,
        height=220
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Continue →"):
        st.session_state.concern = concern
        st.session_state.page = "category"
        st.rerun()

# =========================================================
# CATEGORY PAGE
# =========================================================

elif st.session_state.page == "category":

    st.markdown(
        "<div class='question-title'>What sustainability category interests you?</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='question-subtitle'>This allows sustainability project ideas align to your focus areas.</div>",
        unsafe_allow_html=True
    )

    category = st.selectbox(
        "Select sustainability category",
        categories
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Continue →"):
        st.session_state.category = category
        st.session_state.page = "diploma"
        st.rerun()

# =========================================================
# DIPLOMA PAGE
# =========================================================

elif st.session_state.page == "diploma":

    st.markdown(
        "<div class='question-title'>What is your diploma?</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='question-subtitle'>This helps me tailor sustainability project ideas to your field of study.</div>",
        unsafe_allow_html=True
    )

    diploma = st.selectbox(
        "Select your diploma",
        diplomas
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Generate Ideas"):
        st.session_state.diploma = diploma

        prompt = f"""
        Generate 3 sustainability project ideas.

        Requirements:
        - Tailored to the diploma
        - Related to sustainability category
        - Match the solution type
        - Practical and achievable for diploma students
        - Innovative but realistic
        - Around 120-160 words each

        Diploma:
        {st.session_state.diploma}

        Sustainability Category:
        {st.session_state.category}

        Sustainability Concern:
        {st.session_state.concern}

        Solution Type:
        {st.session_state.solution_type}

        Format:
        1. Title
        Description

        2. Title
        Description

        3. Title
        Description
        """

        response = model.generate_content(prompt)

        ideas = response.text.split("\n\n")

        st.session_state.ideas = ideas
        st.session_state.idea_index = 0
        st.session_state.page = "result"

        st.rerun()

# =========================================================
# RESULT PAGE
# =========================================================

elif st.session_state.page == "result":

    st.markdown(
        "<div class='result-title'>Here are your</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class='result-heading'>
        Project <span style='color:#A8C256;'>Ideas!</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    current_idea = st.session_state.ideas[st.session_state.idea_index]

    st.markdown(
        f"""
        <div class='idea-box'>
            <div class='idea-content'>
                {current_idea.replace('\n', '<br>')}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,1,1])

    with col1:
        if st.button("⬅ Previous"):
            if st.session_state.idea_index > 0:
                st.session_state.idea_index -= 1
                st.rerun()

    with col2:
        if st.button("Start Over"):
            st.session_state.page = "welcome"
            st.rerun()

    with col3:
        if st.button("Next ➡"):
            if st.session_state.idea_index < len(st.session_state.ideas) - 1:
                st.session_state.idea_index += 1
                st.rerun()
