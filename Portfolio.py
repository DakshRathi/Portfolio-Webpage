import streamlit as st
from streamlit_lottie import st_lottie
import requests

# Set page config for wide layout and persistent scroll
st.set_page_config(page_title="Daksh's Portfolio", layout="wide")

# Custom CSS for smooth scrolling and section dividers
st.markdown("""
    <style>
        html, body, [data-testid="stAppViewContainer"] {
            scroll-behavior: smooth;
        }
        
        .section {
            padding-top: 80px;
        }
        
        /* Sidebar links styling */
        [data-testid="stSidebar"] a {
            color: inherit !important;
            text-decoration: none !important;
            font-size: 16px;
            font-weight: bold;
            display: block;
            padding: 8px;
            border-radius: 4px;
        }

        [data-testid="stSidebar"] a:hover {
            background-color: lightblue;
        }

        [data-testid="stSidebar"] ul {
            list-style-type: none;
            padding-left: 0;
        }

        /* Colorful section dividers */
        .divider {
        height: 2px;
        width: 100%;
        background: linear-gradient(90deg, red, orange, yellow, green, blue, indigo, violet);
        display: block;
    }

     /* Light mode styles */
    @media (prefers-color-scheme: light) {
        [data-testid="stSidebar"] {
            color: black !important;
            background-color: #f0f2f6 !important; /* Default light mode background */
        }
        [data-testid="stSidebar"] a {
            color: black !important;
        }
    }

    /* Dark mode styles */
    @media (prefers-color-scheme: dark) {
        [data-testid="stSidebar"] {
            color: white !important;
            background-color: #262730 !important; /* Default dark mode background */
        }
        [data-testid="stSidebar"] a {
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

def add_color_bar():
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_intro = load_lottie_url("https://lottie.host/c35f9be8-9718-4cc4-92d8-31f2e14add72/9TWnyruoCm.json")
lottie_skills = load_lottie_url("https://lottie.host/09fff519-33f1-4906-80e9-7b1f13b239b1/VQ4bYbhu8E.json")

# Sidebar navigation with direct jump links
st.sidebar.title('Table of Contents')
st.sidebar.markdown("""
    - [Introduction](#introduction)
    - [Projects](#projects)
    - [Skills](#skills)
    - [AI Buddy](#ai-buddy)
    - [Let's Connect](#connect)
""", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div style="display: flex; align-items: center;">'
    '<img src="https://cdn-icons-png.flaticon.com/512/2991/2991108.png" width="20" style="margin-right: 10px;"/>'
    '<a href="https://drive.google.com/uc?export=download&id=1ima7zJp4G3a2kDgq9EHInfelPuQa45Qs" target="_blank" style="text-decoration: none; font-weight: bold;">Download Resume</a>'
    '</div>',
    unsafe_allow_html=True
)

# Introduction Section
st.markdown('<h2 id="introduction" class="section">Introduction</h2>', unsafe_allow_html=True)
add_color_bar()
with st.container():
    col1,col2 = st.columns([5,3])

    with col1:
        st.markdown("""
        Hello,  
        ## **I'm Daksh Rathi**
        **IIT (BHU) Varanasi '26 B.Tech Student**  

        I am passionate about **Artificial Intelligence, Machine Learning and Software Development**. I focus on bridging the gap between **theory and real-world applications** by leveraging data-driven insights and AI-powered solutions.  

        My expertise includes **ML deployment, MLOps, and full-stack AI development**, with hands-on experience in **cloud computing, model optimization, and automation**.  
        I am committed to building scalable, impactful, and efficient technology solutions. 🚀  
        """)

    with col2:
        st.lottie(lottie_intro, speed=1, height=200, key="intro")

# Projects Section
st.markdown('<h2 id="projects" class="section">Projects</h2>', unsafe_allow_html=True)
add_color_bar()

# Project 1: YouTube Comment Sentiment Analysis Plugin
with st.container():
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("assets/YouTubeCommentAnalyzer.png", caption="YouTube Sentiment Analysis Plugin", use_container_width=True)
    with col2:
        st.markdown("### **YouTube Comment Sentiment Analysis Plugin**")
        st.write("""
        - Built a Chrome extension for analyzing YouTube comments, offering sentiment classification (positive, neutral, negative), AI-based comment summaries, trend graphs and word clouds
        - Used **FastAPI, LightGBM, TF-IDF** for real-time sentiment classification and improved model accuracy from 60% to 86% through hyperparameter tuning with **Optuna**
        - Managed experiments using **MLflow and Dagshub**, while employing **DVC** for data/model versioning and an automated training pipeline
        - Deployed the backend on **AWS EC2** with **Docker** and **AWS ECR**, integrated **CI/CD** pipelines using **GitHub Actions**, and utilized **AWS CodeDeploy** with Auto Scaling Groups and Elastic Load Balancers for scalability and high availability
        """)

# Project 2: Real Estate Analytics and Recommender System
with st.container():
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("### **Real Estate Analytics and Recommender System**")
        st.write("""
        
        - Developed a property price prediction and recommendation system for Gurgaon real estate using Python, Pandas, NumPy, Plotly, and Streamlit. Conducted data preprocessing, exploratory data analysis, feature engineering, and outlier detection to enhance data quality and model performance
        -  Used **XGBoost** to achieve increase in R² score from **73.63% to 90.7%** and reduced mean absolute error in prediction from **94.63 to 45 lakhs**, improving model accuracy and reliability of price predictions. Also implemented a content-based recommender system using cosine similarity for property recommendations
        - Deployed on **AWS EC2** with load balancers & auto-scaling
        """)
    with col2:
        st.image("assets/RealEstateApp.png", caption="Real Estate Analytics & Recommender App", use_container_width=True)

# Project 3: Fine-tuning ModernBERT on Fake News Dataset
with st.container():
    col1, col2 = st.columns([2, 3])
    with col1:
            st.image("assets/FakeNewsThumbnail.jpg", caption="ModernBERT-FakeNewsClassifier", use_container_width=True)
    with col2:
        st.markdown("### **Fine-tuning ModernBERT on Fake News Dataset**")
        st.write("""
        - Fine-tuned **ModernBERT** on a fake news dataset caontaining over 30,000 fake and real news articles for accurate detection.
        - Applied **transfer learning & mixed precision training**.
        - Achieved **high classification accuracy** with advanced fine-tuning.
        """)

# Skills Section
st.markdown('<h2 id="skills" class="section">Skills</h2>', unsafe_allow_html=True)
add_color_bar()

with st.container():
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("#### **Libraries and Frameworks**")
        st.write("Pandas, NumPy, Matplotlib, Seaborn, TensorFlow, Keras, Scikit-learn, XGBoost, PyTorch, FastAI, FastAPI, LightGBM, C++")

        st.markdown("#### **Tools and Technologies**")
        st.write("SQL, MongoDB, Git & GitHub, Docker, DVC, Dagshub, MLflow, AWS, CI/CD")

    with col2:
        st.lottie(lottie_skills, speed=1, height=250, key="skills_animation")

# AI Buddy Section
st.markdown('<h2 id="ai-buddy" class="section">AI Buddy</h2>', unsafe_allow_html=True)
add_color_bar()
st.write(""" To know more about me, interact with my AI Buddy, a chatbot designed to answer your questions and provide insights about my work and experiences""")

# Contact Section
st.markdown('<h2 id="connect" class="section">Let\'s Connect</h2>', unsafe_allow_html=True)
add_color_bar()

col1, col2, col3, _, _= st.columns(5)

with col1:
    st.markdown(
        '<div style="display: flex; align-items: center;">'
        '<a href="https://www.linkedin.com/in/dakshrathi/" target="_blank">'
        '<img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="40" style="margin-right: 10px;"/>'
        '</a>'
        '<span>LinkedIn</span>'
        '</div>', 
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        '<div style="display: flex; align-items: center;">'
        '<a href="https://github.com/DakshRathi" target="_blank">'
        '<img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="40" style="margin-right: 10px;"/>'
        '</a>'
        '<span>GitHub</span>'
        '</div>', 
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        '<div style="display: flex; align-items: center;">'
        '<a href="mailto:dakshvandanarathi@gmail.com">'
        '<img src="https://cdn-icons-png.flaticon.com/512/732/732200.png" width="40" style="margin-right: 10px;"/>'
        '</a>'
        '<span>Email</span>'
        '</div>', 
        unsafe_allow_html=True
    )