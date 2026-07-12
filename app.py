import os
import time
import requests
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Streamlit Page Configuration
st.set_page_config(
    page_title="Career Compass AI",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Blue/White Theme & Cards
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Outfit', sans-serif;
        background-color: #F8FAFC;
    }
    
    /* Hero Section Gradient Container */
    .hero-container {
        text-align: center;
        padding: 2.5rem 1.5rem;
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 15px -3px rgba(30, 58, 138, 0.2);
    }
    
    .hero-icon {
        font-size: 3.5rem;
        margin-bottom: 0.5rem;
    }
    
    .hero-title {
        color: white !important;
        margin: 0;
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }
    
    .hero-desc {
        color: #E2E8F0;
        font-size: 1.05rem;
        max-width: 650px;
        margin: 0.5rem auto 0;
        font-weight: 300;
        line-height: 1.5;
    }
    
    /* Sidebar Details */
    .sidebar-title {
        font-weight: 700;
        color: #1E3A8A;
        font-size: 1.25rem;
        margin-bottom: 0.5rem;
    }
    
    .badge-ibm {
        background: linear-gradient(135deg, #0F4C81 0%, #1E3A8A 100%);
        color: white;
        padding: 0.6rem;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 700;
        text-align: center;
        margin: 1rem 0;
        border-left: 5px solid #00D2FF;
        box-shadow: 0 4px 6px -1px rgba(15, 76, 129, 0.2);
    }
    
    .tech-tag {
        display: inline-block;
        background-color: #EFF6FF;
        color: #1E40AF;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 0.2rem;
        border: 1px solid #DBEAFE;
    }
    
    /* Results Highlight Header Cards */
    .rec-card {
        background: #FFFFFF;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 6px solid #10B981;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    
    .rec-lbl {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        color: #64748B;
        margin-bottom: 0.2rem;
    }
    
    .rec-val {
        font-size: 1.5rem;
        font-weight: 800;
        color: #1E3A8A;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State navigation and data keys
if "screen" not in st.session_state:
    st.session_state["screen"] = "home"

# --- IBM Watsonx.ai Integration Functions ---

def get_ibm_token(api_key: str):
    """Exchanges an IBM Cloud API Key for an IAM bearer access token."""
    auth_url = "https://iam.cloud.ibm.com/identity/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": api_key}
    try:
        response = requests.post(auth_url, headers=headers, data=data, timeout=15)
        if response.status_code == 200:
            return response.json().get("access_token"), None
        return None, "Invalid API Key. Please verify your IBM Cloud credentials."
    except requests.exceptions.Timeout:
        return None, "Connection timed out during authentication."
    except requests.exceptions.RequestException as e:
        return None, f"Network error during authentication: {str(e)}"

def call_granite_chat(api_key: str, project_id: str, region_url: str, prompt: str):
    """Invokes the IBM Granite model using watsonx.ai Chat REST API."""
    token, err = get_ibm_token(api_key)
    if err:
        return None, err
    
    chat_url = f"{region_url.rstrip('/')}/ml/v1/text/chat?version=2024-10-08"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json", "Accept": "application/json"}
    payload = {
        "project_id": project_id,
        "model_id": "ibm/granite-3-8b-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "parameters": {"max_new_tokens": 3000, "temperature": 0.2, "decoding_method": "greedy"}
    }
    
    try:
        response = requests.post(chat_url, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            res_json = response.json()
            if "choices" in res_json and len(res_json["choices"]) > 0:
                return res_json["choices"][0]["message"]["content"], None
            return None, "API returned an unexpected response structure."
        elif response.status_code == 400:
            err_msg = response.json().get("errors", [{}])[0].get("message", "")
            if "project_id" in err_msg.lower():
                return None, "Invalid Project ID. Please verify your watsonx.ai Project ID."
            return None, f"Bad Request: {err_msg or response.text}"
        return None, f"IBM watsonx API error (HTTP {response.status_code})."
    except requests.exceptions.Timeout:
        return None, "The request timed out. IBM watsonx took too long to respond."
    except requests.exceptions.RequestException as e:
        return None, f"Network error connecting to IBM watsonx.ai: {str(e)}"

# --- Helper to parse tagged output ---

def parse_roadmap(text: str) -> dict:
    """Parses the generated response based on specific section tags."""
    sections = {
        "best_career": "", "why_matches": "", "skill_gap": "", "tech_to_learn": "",
        "soft_to_improve": "", "roadmap_30": "", "certifications": "", "projects": "",
        "interview_tips": "", "resume_tips": "", "future_scope": "", "motivational_advice": ""
    }
    tags = {
        "best_career": "[1. CAREER RECOMMENDATION]",
        "why_matches": "[2. WHY THIS CAREER MATCHES THE STUDENT]",
        "skill_gap": "[3. SKILL GAP ANALYSIS]",
        "tech_to_learn": "[4. TECHNICAL SKILLS TO LEARN]",
        "soft_to_improve": "[5. SOFT SKILLS TO IMPROVE]",
        "roadmap_30": "[6. 30-DAY LEARNING ROADMAP]",
        "certifications": "[7. RECOMMENDED CERTIFICATIONS]",
        "projects": "[8. RECOMMENDED PROJECTS]",
        "interview_tips": "[9. INTERVIEW PREPARATION TIPS]",
        "resume_tips": "[10. RESUME IMPROVEMENT TIPS]",
        "future_scope": "[11. FUTURE SCOPE]",
        "motivational_advice": "[12. MOTIVATIONAL ADVICE]"
    }
    
    indices = []
    for key, tag in tags.items():
        pos = text.find(tag)
        if pos != -1:
            indices.append((key, pos, len(tag)))
            
    indices.sort(key=lambda x: x[1])
    
    for i in range(len(indices)):
        key, pos, tag_len = indices[i]
        start_idx = pos + tag_len
        end_idx = indices[i+1][1] if i + 1 < len(indices) else len(text)
        sections[key] = text[start_idx:end_idx].strip()
        
    return sections

def format_txt_roadmap(sections: dict, info: dict, mode: str) -> str:
    """Formats the roadmap sections into a clean text download format."""
    lines = [
        "="*55,
        "               CAREER COMPASS AI ROADMAP",
        f"     Agentic Career Counseling Companion ({'IBM Granite' if mode == 'granite' else 'Demo Mode fallback'})",
        "="*55,
        f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "STUDENT PROFILE:",
        f"- Name: {info['name']}",
        f"- Education: {info['education']}",
        f"- Current Year: {info['year']}",
        f"- Skills: {info['skills']}",
        f"- Interests: {info['interests']}",
        f"- Career Goal: {info['career_goal']}",
        f"- Preferred Industry: {info['industry']}",
        f"- Experience Level: {info['experience_level']}",
        "="*55,
        ""
    ]
    
    labels = {
        "best_career": "1. CAREER RECOMMENDATION",
        "why_matches": "2. WHY THIS CAREER MATCHES THE STUDENT",
        "skill_gap": "3. SKILL GAP ANALYSIS",
        "tech_to_learn": "4. TECHNICAL SKILLS TO LEARN",
        "soft_to_improve": "5. SOFT SKILLS TO IMPROVE",
        "roadmap_30": "6. 30-DAY LEARNING ROADMAP",
        "certifications": "7. RECOMMENDED CERTIFICATIONS",
        "projects": "8. RECOMMENDED PROJECTS",
        "interview_tips": "9. INTERVIEW PREPARATION TIPS",
        "resume_tips": "10. RESUME IMPROVEMENT TIPS",
        "future_scope": "11. FUTURE SCOPE",
        "motivational_advice": "12. MOTIVATIONAL ADVICE"
    }
    
    for key, label in labels.items():
        lines.append(f"[{label}]")
        lines.append(sections.get(key, "N/A"))
        lines.append("\n" + "-"*35 + "\n")
        
    return "\n".join(lines)

# --- Predefined local Demo Mode templates ---

def generate_demo_roadmap(name: str, education: str, year: str, skills_str: str, interests_str: str, career_goal: str, industry: str, experience_level: str) -> dict:
    """Generates a high-quality, realistic mock career roadmap based on student preferences."""
    skills_list = [s.strip() for s in skills_str.split(",") if s.strip()]
    interests_list = [i.strip() for i in interests_str.split(",") if i.strip()]
    goal = career_goal.strip() if career_goal.strip() else "Technology Professional"
    ind = industry.strip() if industry.strip() else "Digital Tech"
    
    goal_lower = goal.lower()
    
    role_mapping = {
        "data scientist": {
            "tech": ["Python Programming", "SQL Database Queries", "Data Analysis (Pandas/NumPy)", "Machine Learning (Scikit-Learn)", "Data Visualization (Tableau/Power BI)", "Model Deployment basics"],
            "certs": ["IBM Data Science Professional Certificate", "Google Data Analytics Certificate", "Microsoft Certified: Azure Data Scientist Associate"],
            "projects": [
                "**1. Predictive Housing Price Model**: A regression model utilizing Python and Scikit-Learn to estimate real estate prices based on geographical and structural indicators.",
                "**2. E-commerce Customer Segmentation Dashboard**: A clustering application grouping customer demographics with K-means, visualized dynamically using Tableau.",
                "**3. Product Sentiment Analyzer API**: A FastAPI service performing Natural Language Processing on scraped product reviews to yield real-time sentiment scoring."
            ]
        },
        "data analyst": {
            "tech": ["Advanced Microsoft Excel", "SQL Databases", "Tableau or Power BI Desktop", "Basic Python/R for analytics", "Statistical Hypothesis Testing", "Reporting & Presentation"],
            "certs": ["Google Data Analytics Professional Certificate", "Microsoft Certified: Power BI Data Analyst Associate", "CompTIA Data+"],
            "projects": [
                "**1. Interactive Retail Performance Dashboard**: A comprehensive Power BI dashboard analyzing sales growth, customer retention, and regional returns.",
                "**2. SQL Database Query Optimizer**: A collection of structured query scripts analyzing employee databases to identify organizational trends and optimize query performance.",
                "**3. Web-scraped Competitor Pricing Tracker**: A Python-based script extracting weekly prices from e-commerce sites to report variance metrics."
            ]
        },
        "software engineer": {
            "tech": ["Data Structures & Algorithms", "Version Control (Git/GitHub)", "System Architecture & OOP", "Unit Testing & Mocking", "RESTful API Integration", "CI/CD Pipelines"],
            "certs": ["AWS Certified Developer - Associate", "Google Professional Cloud Developer", "Oracle Certified Professional: Java SE Developer"],
            "projects": [
                "**1. Scalable Task Management Microservice**: A RESTful API built in Node.js or Spring Boot, featuring PostgreSQL database integration, user validation, and unit tests.",
                "**2. Distributed Log Monitoring Service**: A Go or Python system aggregating terminal output from servers, saving data using custom indexing formats.",
                "**3. Automated Cloud Testing Suite**: A GitHub Actions workflow executing automated test runs on mock server endpoints upon every commit pull."
            ]
        },
        "web developer": {
            "tech": ["HTML5, CSS3, & Modern JavaScript", "Frontend Frameworks (React, Vue, or Angular)", "Backend Development (Node.js/Express)", "Responsive Design & CSS Flexbox/Grid", "Database Management (MongoDB/SQL)", "Package Managers (npm/yarn)"],
            "certs": ["Meta Front-End Developer Professional Certificate", "freeCodeCamp Full Stack Developer Certification", "W3Schools Full Stack Developer Certification"],
            "projects": [
                "**1. Responsive Portfolio Portal**: A personal developer webpage showcasing clean layout structures, custom styling, animations, and form validation.",
                "**2. Real-Time Chat Application**: A full-stack web application leveraging React and Socket.io for immediate peer-to-peer message synchronization.",
                "**3. SaaS Product Landing Page**: A landing platform styled with Tailwind CSS, utilizing Stripe API mock configurations for secure user checkouts."
            ]
        },
        "cloud engineer": {
            "tech": ["Cloud Architecture Basics (AWS/Azure/GCP)", "Linux/Bash Scripting & Automation", "Containerization (Docker)", "Orchestration (Kubernetes)", "Infrastructure as Code (Terraform)", "CI/CD workflows (GitHub Actions/Jenkins)"],
            "certs": ["AWS Certified Cloud Practitioner", "Google Cloud Associate Cloud Engineer", "Microsoft Certified: Azure Administrator Associate"],
            "projects": [
                "**1. Auto-scaling Kubernetes Web Server**: A cloud environment using Docker images deployed to a local Minikube cluster with configured load balancers.",
                "**2. Terraform Cloud Architecture**: Declarative infrastructure scripts setting up VPCs, security groups, and server instances on AWS automatically.",
                "**3. Jenkins Automated Deployment Pipeline**: A full pipeline rebuilding and copying containerized updates to staging machines after testing."
            ]
        },
        "cybersecurity": {
            "tech": ["Network Security & Protocols", "Linux Administration", "Ethical Hacking / Penetration Testing", "Security Information and Event Management (SIEM)", "Vulnerability Assessment Tools (Nmap/Nessus)", "Cryptography basics"],
            "certs": ["CompTIA Security+", "Certified Information Systems Security Professional (CISSP)", "Certified Ethical Hacker (CEH)"],
            "projects": [
                "**1. Network Packet Analyzer Tool**: A custom Python application using Scapy to sniff and report anomalies in local Wireshark telemetry.",
                "**2. Automated Port Scanner & Reporter**: A Bash automation utility testing open ports across a range of subnet hosts and drafting security summaries.",
                "**3. Intrusions Detection Alert Microservice**: An endpoint script sending immediate Discord notifications when SSH logs detect unauthorized access."
            ]
        }
    }
    
    # Identify best matching role in templates
    matched_role = None
    for key in role_mapping:
        if key in goal_lower:
            matched_role = role_mapping[key]
            break
            
    if not matched_role:
        # Default fallback role template
        matched_role = {
            "tech": ["Core Programming Language (Python/Java/JS)", "Relational Databases & SQL", "Git & Version Control", "REST API Development", "Cloud Deployment Basics (AWS/Render)", "Unit Testing Frameworks"],
            "certs": ["AWS Certified Cloud Practitioner", "Google Professional IT Certificate", "IBM Professional Core Certificate"],
            "projects": [
                "**1. End-to-End Core Logic Prototype**: A backend application demonstrating clear structure, input validation, and clean logging setups.",
                "**2. Relational Database Dashboard**: A data-driven system mapping tables, running queries, and displaying visual summary charts.",
                "**3. Cloud Deployed Application**: A lightweight web application configured with automated tests and hosted on a public cloud server."
            ]
        }
        
    # Determine missing skills
    missing_tech = [t for t in matched_role["tech"] if not any(existing.lower() in t.lower() for existing in skills_list)]
    if not missing_tech:
        missing_tech = ["Advanced Optimization Techniques", "System Scalability Design", "Automated DevOps Pipelines"]
        
    sections = {}
    
    # 1. Career Recommendation
    sections["best_career"] = f"Junior {goal} (Recommended Experience Track: {experience_level}) in the {ind} industry"
    
    # 2. Why this career matches the student
    sections["why_matches"] = (
        f"Your academic profile (**{education}**, status: **{year}**) forms a great launching pad for a career as a **{goal}**. "
        f"Your specific interest in *{', '.join(interests_list) if interests_list else 'modern technology'}* shows a strong curiosity for the core problems this role solves. "
        f"Furthermore, your existing skills in *{', '.join(skills_list) if skills_list else 'problem solving'}* provide a useful baseline. "
        f"By pursuing this career pathway, you will be able to align your day-to-day work directly with your ambitions, "
        f"contributing high-value solutions to the **{ind}** sector."
    )
    
    # 3. Skill Gap Analysis
    sections["skill_gap"] = (
        f"An analysis of your current profile shows that while you possess solid foundational knowledge, transitioning into a **{experience_level}** {goal} "
        f"requires building additional skills. To stand out, you must upgrade your skills from academic assignments "
        f"to production-level development. The gap is primarily in technical tools like **{missing_tech[0]}** "
        f"and engineering practices like version control, testing, and cloud environments."
    )
    
    # 4. Technical Skills to Learn
    sections["tech_to_learn"] = "\n".join([f"- **{t}**: Crucial tool required to build modern applications in the {ind} industry." for t in missing_tech])
    
    # 5. Soft Skills to Improve
    sections["soft_to_improve"] = (
        "- **Technical Communication**: Translating code patterns or mathematical insights into clear business requirements.\n"
        "- **Collaboration & Git Etiquette**: Working effectively in standard pull-request/code-review teams.\n"
        "- **Agile Prioritization**: Managing tasks and adjusting timelines in sprints."
    )
    
    # 6. 30-Day Learning Roadmap (Week 1 to Week 4)
    sections["roadmap_30"] = (
        f"### 📅 Week 1: Core Technologies Strengthening\n"
        f"- **Objective**: Build deep familiarity with **{missing_tech[0]}**.\n"
        f"- **Actions**: Study theoretical basics, set up your development environment, and write small utility scripts. Aim for 2 hours of hands-on practice daily.\n\n"
        f"### 📅 Week 2: Intermediate Tools & Frameworks\n"
        f"- **Objective**: Master **{missing_tech[1] if len(missing_tech) > 1 else 'Database Systems & Version Control'}**.\n"
        f"- **Actions**: Practice querying, learn git branching, merge pull requests locally, and run simple automated tests.\n\n"
        f"### 📅 Week 3: Capstone Project Construction\n"
        f"- **Objective**: Apply your learnings by building a personal project related to the **{ind}** industry.\n"
        f"- **Actions**: Code the application, write unit tests, design the schema, and document it with a clean README file on GitHub.\n\n"
        f"### 📅 Week 4: Cloud Infrastructure & Staging\n"
        f"- **Objective**: Deployment, hosting, and portfolio integration.\n"
        f"- **Actions**: Deploy the capstone prototype onto a public platform (e.g. Render, Vercel, or Netlify). Optimize code formatting, format your resume, and conduct mock interviews."
    )
    
    # 7. Recommended Certifications
    sections["certifications"] = (
        "**Online Courses:**\n"
        f"- **Coursera / edX {goal} Fundamentals**: Focuses on basic syntax and structural designs.\n"
        f"- **Udemy High-Performance {goal} Masterclass**: Highlights architecture patterns.\n\n"
        "**Certifications:**\n" +
        "\n".join([f"- **{cert}**" for cert in matched_role["certs"]])
    )
    
    # 8. Recommended Projects
    sections["projects"] = "\n\n".join(matched_role["projects"])
    
    # 9. Interview Preparation Tips
    sections["interview_tips"] = (
        "- **Solve Technical Coding Problems**: Dedicate time daily to data structure practices or database syntax exercises.\n"
        "- **Study System Architecture**: Understand how servers, databases, and frontend interfaces interact over REST or WebSockets.\n"
        "- **Practice Behavioral Questions**: Format your answers using the STAR format (Situation, Task, Action, Result) to show problem-solving skills under stress."
    )
    
    # 10. Resume Improvement Tips
    sections["resume_tips"] = (
        "- **Use Metrics and Action Verbs**: Write 'Created responsive API handling 5,000 requests daily' instead of 'Built basic API'.\n"
        "- **Highlight Relevant Skills**: List *{', '.join(skills_list)}* prominently in a dedicated technical section.\n"
        "- **Showcase Active Projects**: Place a dedicated projects section at the top of your resume, including direct hyperlinks to active live sites."
    )
    
    # 11. Future Scope
    sections["future_scope"] = (
        f"The demand for a skilled **{goal}** inside the **{ind}** industry continues to grow rapidly. "
        f"With companies shifting towards cloud migration, automation, and AI analytics, specialized engineers are in high demand. "
        f"Typical career progression starts at junior developer, moving to Senior Architect, Lead Developer, and Technical Manager within 4-7 years."
    )
    
    # 12. Motivational Advice
    sections["motivational_advice"] = (
        f"Dear {name}, success in tech is not about knowing everything, but demonstrating a persistent capacity to learn. "
        f"Do not be discouraged by initial bugs or rejected applications. "
        f"Building skills step-by-step using this roadmap will distinguish you from other candidates. "
        f"Take the first step today by writing code, building projects, and updating your profile. "
        f"Your dream career as a {goal} is within your reach!"
    )
    
    return sections

# --- Sidebar Configuration ---

st.sidebar.markdown('<div class="sidebar-title">🧭 Career Compass AI</div>', unsafe_allow_html=True)
st.sidebar.markdown(
    "Career Compass AI is an agentic counseling companion designed to align your academic credentials, "
    "skills, and interests with the best-fit industry pathway using IBM Granite AI."
)

st.sidebar.markdown('<div class="badge-ibm">🤖 Powered by IBM Granite</div>', unsafe_allow_html=True)

st.sidebar.markdown("### 🛠️ Technologies Used")
st.sidebar.markdown("""
<span class="tech-tag">Python</span>
<span class="tech-tag">Streamlit</span>
<span class="tech-tag">IBM watsonx.ai</span>
<span class="tech-tag">Granite 3.0 Instruct</span>
<span class="tech-tag">Requests</span>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔑 IBM watsonx.ai Credentials")

# Load environment credentials
env_apikey = os.getenv("WATSONX_APIKEY", "")
env_project = os.getenv("WATSONX_PROJECT_ID", "")
env_url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

sidebar_api_key = st.sidebar.text_input("IBM Cloud API Key", value=env_apikey, type="password", help="Obtain from IBM Cloud IAM console.")
sidebar_project_id = st.sidebar.text_input("watsonx.ai Project ID", value=env_project, help="Obtain from watsonx.ai Project Manage settings.")
sidebar_url = st.sidebar.selectbox("watsonx.ai Region URL", [
    "https://us-south.ml.cloud.ibm.com",
    "https://eu-de.ml.cloud.ibm.com",
    "https://ap-south.ml.cloud.ibm.com"
], index=["https://us-south.ml.cloud.ibm.com", "https://eu-de.ml.cloud.ibm.com", "https://ap-south.ml.cloud.ibm.com"].index(env_url) if env_url in ["https://us-south.ml.cloud.ibm.com", "https://eu-de.ml.cloud.ibm.com", "https://ap-south.ml.cloud.ibm.com"] else 0)

# --- Navigation Screen Routing ---

if st.session_state["screen"] == "home":
    # SCREEN 1: HOME SCREEN
    
    # Hero Header Section
    st.markdown("""
    <div class="hero-container">
        <div class="hero-icon">🧭</div>
        <h1 class="hero-title">Career Compass AI</h1>
        <p class="hero-desc">
            An agentic career counseling companion. Submit your academic credentials and interests below, and let the AI agent chart your personalized professional roadmap.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Retrieve form persistence values from session state
    val_name = st.session_state.get("form_name", "")
    val_education = st.session_state.get("form_education", "Bachelor's Degree")
    val_year = st.session_state.get("form_year", "1st Year")
    val_experience = st.session_state.get("form_experience", "Beginner")
    val_skills = st.session_state.get("form_skills", "")
    val_interests = st.session_state.get("form_interests", "")
    val_goal = st.session_state.get("form_goal", "")
    val_industry = st.session_state.get("form_industry", "")
    
    # Main Profile Input Form
    with st.form("student_profile_form"):
        st.markdown("### 👤 Student Profile & Preferences")
        
        col1, col2 = st.columns(2)
        with col1:
            full_name = st.text_input("Full Name", value=val_name, placeholder="e.g., Jane Doe")
            education_list = ["High School", "Diploma / Vocational Degree", "Bachelor's Degree", "Master's Degree", "Doctorate (Ph.D.)"]
            education = st.selectbox("Education Level", education_list, index=education_list.index(val_education) if val_education in education_list else 2)
            
            year_list = ["1st Year", "2nd Year", "3rd Year", "4th Year", "Graduated / Working Professional"]
            current_year = st.selectbox("Current Academic Year / Status", year_list, index=year_list.index(val_year) if val_year in year_list else 0)
            
            exp_list = ["Beginner", "Intermediate", "Advanced"]
            experience_level = st.selectbox("Career Experience Level", exp_list, index=exp_list.index(val_experience) if val_experience in exp_list else 0)
            
        with col2:
            skills = st.text_input("Current Skills", value=val_skills, placeholder="e.g., Python, Excel, SQL, Public Speaking")
            interests = st.text_input("Interests & Hobbies", value=val_interests, placeholder="e.g., Web Development, Finance, Machine Learning")
            career_goal = st.text_input("Career Goal / Target Role", value=val_goal, placeholder="e.g., Become a Data Scientist")
            industry = st.text_input("Preferred Industry", value=val_industry, placeholder="e.g., Healthcare, FinTech, E-commerce")

        # Form Submission buttons
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, _ = st.columns([2, 1.5, 5])
        with c1:
            generate_btn = st.form_submit_button("Generate Career Roadmap", use_container_width=True)
        with c2:
            clear_btn = st.form_submit_button("Reset Form", use_container_width=True)

    # Handle Reset Form action (wipes both credentials/inputs data and routing)
    if clear_btn:
        for k in ["raw_roadmap", "parsed_roadmap", "student_info", "mode", "form_name", "form_education", "form_year", "form_experience", "form_skills", "form_interests", "form_goal", "form_industry"]:
            if k in st.session_state:
                del st.session_state[k]
        st.session_state["screen"] = "home"
        st.rerun()

    # Run logic on Generate click
    if generate_btn:
        # Check if credentials are present to determine mode
        use_granite = bool(sidebar_api_key.strip() and sidebar_project_id.strip())
        
        # 1. Inputs Validation
        if not full_name.strip():
            st.error("⚠️ Please fill in your Full Name to proceed.")
        elif not skills.strip():
            st.error("⚠️ Please list at least one current Skill.")
        elif not interests.strip():
            st.error("⚠️ Please enter at least one Interest.")
        elif not career_goal.strip():
            st.error("⚠️ Please specify a Career Goal / Target Role.")
        elif not industry.strip():
            st.error("⚠️ Please select your Preferred Industry.")
        else:
            # Persistent form memory state saving
            st.session_state["form_name"] = full_name.strip()
            st.session_state["form_education"] = education
            st.session_state["form_year"] = current_year
            st.session_state["form_experience"] = experience_level
            st.session_state["form_skills"] = skills.strip()
            st.session_state["form_interests"] = interests.strip()
            st.session_state["form_goal"] = career_goal.strip()
            st.session_state["form_industry"] = industry.strip()
            
            # Save student information dictionary
            info = {
                "name": full_name.strip(),
                "education": education,
                "year": current_year,
                "skills": skills.strip(),
                "interests": interests.strip(),
                "career_goal": career_goal.strip(),
                "industry": industry.strip(),
                "experience_level": experience_level
            }
            
            # 2. Simulated Agent Sequence (to make it feel like an AI Agent)
            progress_placeholder = st.empty()
            with progress_placeholder.container():
                st.markdown("### 🤖 Career Agent Processing Sequence")
                steps = [
                    ("🔍", "Analyzing profile credentials and academic background..."),
                    ("📊", "Mapping current skills against industry standard competencies..."),
                    ("🧠", "Consulting career counseling database..."),
                    ("🎯", "Conducting gap analysis to identify missing skillsets..."),
                    ("📅", "Synthesizing week-by-week learning roadmap..."),
                    ("🚀", "Assembling projects, certifications, and preparation guides...")
                ]
                
                progress_bar = st.progress(0)
                for idx, (emoji, desc) in enumerate(steps):
                    st.info(f"{emoji} **Agent Status**: {desc}")
                    time.sleep(0.3)
                    progress_bar.progress((idx + 1) / len(steps))
                
                st.toast("Agent sequence complete! Constructing output results...", icon="✨")
                
            progress_placeholder.empty()

            # 3. Generate Roadmap Content
            if use_granite:
                # Build prompt with 12 tags
                prompt = f"""You are an expert AI Career Counselor. Analyze the student's profile and generate a comprehensive, highly personalized career roadmap.

Student Profile:
- Name: {info['name']}
- Education: {info['education']}
- Current Year: {info['year']}
- Current Skills: {info['skills']}
- Interests: {info['interests']}
- Career Goal: {info['career_goal']}
- Preferred Industry: {info['industry']}
- Experience Level: {info['experience_level']}

You MUST generate exactly 12 distinct sections. Each section MUST start with its exact tag on a new line. Do not modify or omit the tags.

Use the following exact format:

[1. CAREER RECOMMENDATION]
Provide the single best career role recommendation (including approximate starting salary range).

[2. WHY THIS CAREER MATCHES THE STUDENT]
Explain how the recommendation matches their skills, education, interests, and career goals.

[3. SKILL GAP ANALYSIS]
Compare the student's current skills against the industry standards for this role, identifying high-priority deficiencies.

[4. TECHNICAL SKILLS TO LEARN]
List the core technical skills and technologies the student must acquire, formatted as bullet points.

[5. SOFT SKILLS TO IMPROVE]
List the interpersonal, communication, or management skills needed, formatted as bullet points.

[6. 30-DAY LEARNING ROADMAP]
Provide a detailed week-by-week plan (Week 1 to Week 4) for acquiring the missing skills and building proficiency.

[7. RECOMMENDED CERTIFICATIONS]
List 2-3 highly valued industry certifications and online courses for this career path.

[8. RECOMMENDED PROJECTS]
Describe 2-3 beginner-friendly project ideas they can build to showcase these skills.

[9. INTERVIEW PREPARATION TIPS]
List key tips, typical questions, and study areas to prepare for interviews in this field.

[10. RESUME IMPROVEMENT TIPS]
List actionable tips to modify their resume, highlight relevant skills, and format projects.

[11. FUTURE SCOPE]
Explain the market demand, growth potential, and career progression roles (future job roles).

[12. MOTIVATIONAL ADVICE]
Provide encouraging words and a positive final message for the student."""

                with st.spinner("Retrieving final recommendations from IBM Granite AI..."):
                    response_text, api_error = call_granite_chat(
                        sidebar_api_key.strip(),
                        sidebar_project_id.strip(),
                        sidebar_url,
                        prompt
                    )

                if api_error:
                    st.error(f"❌ {api_error}")
                elif not response_text:
                    st.error("❌ Received empty response from IBM Granite LLM. Please try again.")
                else:
                    st.session_state["raw_roadmap"] = response_text
                    st.session_state["parsed_roadmap"] = parse_roadmap(response_text)
                    st.session_state["student_info"] = info
                    st.session_state["mode"] = "granite"
                    st.session_state["screen"] = "results"
                    st.rerun()
            else:
                # Fallback to local Demo Mode
                with st.spinner("Generating customized roadmap via local Demo Engine..."):
                    time.sleep(1.0)
                    roadmap_data = generate_demo_roadmap(
                        info["name"], info["education"], info["year"], info["skills"],
                        info["interests"], info["career_goal"], info["industry"], info["experience_level"]
                    )
                    
                    # Create raw code string representation
                    raw_text = ""
                    labels = {
                        "best_career": "1. CAREER RECOMMENDATION",
                        "why_matches": "2. WHY THIS CAREER MATCHES THE STUDENT",
                        "skill_gap": "3. SKILL GAP ANALYSIS",
                        "tech_to_learn": "4. TECHNICAL SKILLS TO LEARN",
                        "soft_to_improve": "5. SOFT SKILLS TO IMPROVE",
                        "roadmap_30": "6. 30-DAY LEARNING ROADMAP",
                        "certifications": "7. RECOMMENDED CERTIFICATIONS",
                        "projects": "8. RECOMMENDED PROJECTS",
                        "interview_tips": "9. INTERVIEW PREPARATION TIPS",
                        "resume_tips": "10. RESUME IMPROVEMENT TIPS",
                        "future_scope": "11. FUTURE SCOPE",
                        "motivational_advice": "12. MOTIVATIONAL ADVICE"
                    }
                    for k, lbl in labels.items():
                        raw_text += f"[{lbl}]\n{roadmap_data[k]}\n\n"
                    
                    st.session_state["raw_roadmap"] = raw_text
                    st.session_state["parsed_roadmap"] = roadmap_data
                    st.session_state["student_info"] = info
                    st.session_state["mode"] = "demo"
                    st.session_state["screen"] = "results"
                    st.rerun()

elif st.session_state["screen"] == "results":
    # SCREEN 2: RESULTS DASHBOARD SCREEN
    
    # Back to Home Button at the very top
    if st.button("← Back to Home"):
        st.session_state["screen"] = "home"
        st.rerun()
        
    st.success("🎉 Your personalized Career Roadmap has been generated successfully!")
    
    # Retrieve session variables
    r = st.session_state["parsed_roadmap"]
    info = st.session_state["student_info"]
    raw_text = st.session_state["raw_roadmap"]
    mode = st.session_state.get("mode", "demo")
    
    st.markdown("---")
    st.markdown("## 📋 Your AI Career Counsel Results")
    
    if mode == "demo":
        st.info("ℹ️ Running in Demo Mode. IBM Granite API is not configured.")
        
    # Student Profile Summary Dashboard Card
    with st.container(border=True):
        st.markdown("### 🎓 Student Profile Summary")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown(f"👤 **Name**: {info['name']}")
            st.markdown(f"🎓 **Education Level**: {info['education']} ({info['year']})")
            st.markdown(f"⚡ **Experience Track**: {info['experience_level']}")
        with col_s2:
            st.markdown(f"🎯 **Target Goal**: {info['career_goal']}")
            st.markdown(f"💼 **Preferred Industry**: {info['industry']}")
            st.markdown(f"🛠️ **Current Skills**: *{info['skills']}*")
            st.markdown(f"🎨 **Interests**: *{info['interests']}*")
            
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Render the 11 expandable sections in beautiful container-based cards
    
    # 1. Career Recommendation
    with st.expander("🎯 Career Recommendation", expanded=True):
        with st.container(border=True):
            st.markdown(r.get("best_career", "N/A"))
            
    # 2. Why this career fits
    with st.expander("🤝 Why this career fits", expanded=True):
        with st.container(border=True):
            st.markdown(r.get("why_matches", "N/A"))
            
    # 3. Skill Gap Analysis
    with st.expander("📊 Skill Gap Analysis", expanded=True):
        with st.container(border=True):
            st.markdown(r.get("skill_gap", "N/A"))
            
    # 4. Technical Skills
    with st.expander("🛠️ Technical Skills", expanded=True):
        with st.container(border=True):
            st.markdown(r.get("tech_to_learn", "N/A"))
            
    # 5. Soft Skills
    with st.expander("💬 Soft Skills", expanded=True):
        with st.container(border=True):
            st.markdown(r.get("soft_to_improve", "N/A"))
            
    # 6. 30-Day Learning Roadmap
    with st.expander("📅 30-Day Learning Roadmap", expanded=True):
        with st.container(border=True):
            st.markdown(r.get("roadmap_30", "N/A"))
            
    # 7. Recommended Projects
    with st.expander("🚀 Recommended Projects", expanded=True):
        with st.container(border=True):
            st.markdown(r.get("projects", "N/A"))
            
    # 8. Certifications
    with st.expander("🏆 Certifications", expanded=True):
        with st.container(border=True):
            st.markdown(r.get("certifications", "N/A"))
            
    # 9. Interview Tips
    with st.expander("🤝 Interview Tips", expanded=True):
        with st.container(border=True):
            st.markdown(r.get("interview_tips", "N/A"))
            
    # 10. Resume Tips
    with st.expander("📝 Resume Tips", expanded=True):
        with st.container(border=True):
            st.markdown(r.get("resume_tips", "N/A"))
            
    # 11. Final Career Advice (merges Future Scope and Motivational Advice)
    with st.expander("💡 Final Career Advice", expanded=True):
        with st.container(border=True):
            st.markdown("### 📈 Future Scope & Growth Potential")
            st.markdown(r.get("future_scope", "N/A"))
        with st.container(border=True):
            st.markdown("### 💡 Motivational Counseling Advice")
            st.markdown(r.get("motivational_advice", "N/A"))

    # File Download and Raw Code block for easy copying
    st.markdown("---")
    st.markdown("### 💾 Export & Copy Options")
    
    formatted_roadmap = format_txt_roadmap(r, info, mode)
    
    col_d1, col_d2 = st.columns([1, 4])
    with col_d1:
        st.download_button(
            label="💾 Download Roadmap (.TXT)",
            data=formatted_roadmap,
            file_name=f"Career_Compass_Roadmap_{info['name'].replace(' ', '_')}.txt",
            mime="text/plain",
            use_container_width=True
        )
        
    with col_d2:
        with st.expander("📋 Click here to view and copy Raw Markdown"):
            st.code(raw_text, language="markdown")
