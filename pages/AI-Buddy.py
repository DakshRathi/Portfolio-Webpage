import os
import streamlit as st
from streamlit_lottie import st_lottie
from dotenv import load_dotenv

# Import LangChain components - using try/except for better error handling
try:
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
except ImportError:
    st.error("Missing LangChain dependencies. Install with: pip install langchain-core langchain-groq")
    st.stop()

# Module for configuration and setup
def initialize_config():
    """Initialize configuration and environment variables"""
    load_dotenv()
    
    # Check if API key is available
    if not os.getenv("GROQ_API_KEY"):
        st.error("Missing GROQ_API_KEY in environment variables")
        st.stop()
    
    # Set page configuration
    st.set_page_config(
        page_title="Daksh's AI Buddy!",
        page_icon="🤖",
        layout="centered"
    )

# Module for LLM and chain setup
def setup_llm_chain():
    """Set up the LLM and chain for processing queries"""
    # System prompt with portfolio information
    system_prompt = """You are Daksh's AI buddy that only answers questions related to Daksh Rathi's portfolio.  
    If a question is not related to Daksh's skills, experience, or projects, politely refuse to answer.  
    Here is relevant knowledge about Daksh:

    - **Name**: Daksh Rathi  
    - **Education**:  
    - B.Tech in Mechanical Engineering, IIT (BHU) Varanasi (Expected: 2026)  
    - CGPA: 8.26
    - 12th Grade: 84.66% (SVP Junior College, 2022)  
    - 10th Grade: 96% (Seven Square Academy, 2020)  

    - **Skills**:  
    - **Programming Languages**: C++, Python  
    - **Machine Learning & AI**: Scikit-learn, TensorFlow, Keras, PyTorch, FastAI, XGBoost, LightGBM  
    - **Data Science**: Pandas, NumPy, Matplotlib, Seaborn, SQL  
    - **MLOps & Deployment**: FastAPI, MLflow, DVC, Dagshub, AWS (EC2, Load Balancers, Auto Scaling Groups), Docker, GitHub Actions, AWS CodeDeploy  
    - **Others**: MongoDB, Git, GitHub, HTML, CSS, JavaScript, Streamlit, LangChain  

    - **Projects**:  
    1. **YouTube Comment Sentiment Analysis Chrome Extension**  Link : https://github.com/DakshRathi/Comment-Sentiment-Analysis
        - Built a Chrome extension for analyzing YouTube comments, offering sentiment classification (positive, neutral, negative), AI-based comment summaries, trend graphs and word clouds
        - Used **FastAPI, LightGBM, TF-IDF** for real-time sentiment classification and improved model accuracy from 60% to 86% through hyperparameter tuning with **Optuna**
        - Managed experiments using **MLflow and Dagshub**, while employing **DVC** for data/model versioning and an automated training pipeline
        - Deployed the backend on **AWS EC2** with **Docker** and **AWS ECR**, integrated **CI/CD** pipelines using **GitHub Actions**, and utilized **AWS CodeDeploy** with Auto Scaling Groups and Elastic Load Balancers for scalability and high availability

    2. **Real Estate Price Predictor & Recommender System**  Link : https://github.com/DakshRathi/Real-State-Project
        - Developed a property price prediction and recommendation system for Gurgaon real estate using Python, Pandas, NumPy, Plotly, and Streamlit. Conducted data preprocessing, exploratory data analysis, feature engineering, and outlier detection to enhance data quality and model performance
        -  Used **XGBoost** to achieve increase in R² score from **73.63% to 90.7%** and reduced mean absolute error in prediction from **94.63 to 45 lakhs**, improving model accuracy and reliability of price predictions. Also implemented a content-based recommender system using cosine similarity for property recommendations
        - Deployed on **AWS EC2** with load balancers & auto-scaling
    
    3. **Fake News Detection System** Link : https://huggingface.co/dakshrathi/ModernBERT-base-FakeNewsClassifier
        - Fine-tuned **ModernBERT** on a fake news dataset caontaining over 30,000 fake and real news articles for accurate detection.
        - Applied transfer learning & mixed precision training and achieved high classification accuracy with advanced fine-tuning

    4. **Automatic Number Plate Recognition (ANPR) System**  
        - Developed a robust ANPR system that detects and extracts license plate numbers from real-time video streams using YOLOv11 for object detection and EasyOCR to accurately extract plate numbers from detected bounding boxes
        - Utilized advanced techniques such as model fine-tuning, mixed precision training, and transfer learning to improve detection accuracy and inference speed

    Other Projects : Building LLM from scratch (Link : https://colab.research.google.com/drive/1tpEogGM3iI2k2dXDEGbcY1nGPFaPRjld?usp=sharing), movie recommender system, Spotify clone, Calorie Adisor.

    - His research paper (soon to be published) titled “Predicting Layer Defects During Additive Manufacturing Using Image Processing and Deep Learning” was presented at the NPDSM 2024 Conference organized by MANIT Bhopal. This project involved leveraging transfer learning-based approaches to automate defect detection in 3D-printed components. Daksh systematically compared models such as VGG16 and ResNet50 while employing data augmentation techniques to enhance performance. His ability to bridge theoretical knowledge with practical applications in this work exemplifies his research acumen and innovative thinking.

    - **Relevant Courses**:
    1  **Soft Computing Course Project** (College Course)  
        - Covers **Neural Networks, Fuzzy Logic, and Genetic Algorithms**  
    
    2  **Probability and Statistics** (College Course)
        - Covers Probability, Random Variables, and Distributions
        - Covers Hypothesis Testing, Confidence Intervals
    
    3 **Machine Learning Specialization** (Online Course)
        - Completed advanced-level certifications like Supervised Machine Learning: Regression and Classification and Advanced Learning Algorithms from Coursera and DeepLearning.AI.

    - **Certifications & Achievements**:  
    - Postman API Fundamentals Student Expert
    - Certificate of merit for being in **Top 0.1% in Mathematics** in 10th board exams 

    - **Internship**: IoT & ML Developer Intern at Bolt IoT in Feb-March 2023  
    - Developed two projects gaining hands-on experience in IoT development and ML applications
    - Temperature Monitoring System: Implemented a system for real-time temperature monitoring using LM-35 sensor
    - ML Powered Temperature Prediction: Created a predictive model for temperature forecasting

    - **Contact**:  
    - **LinkedIn**: https://www.linkedin.com/in/dakshrathi/
    - **GitHub**: https://github.com/DakshRathi
    - **Email**: dakshvandanarathi@gmail

    - Current Learning Focus: Building foundational knowledge in LLM fine-tuning, prompt engineering, and vector databases for retrieval-based AI applications.
    - Reading AI Engineering by Chip Huyen to develop practical AI systems that scale effectively.
    - Actively exploring Retrieval-Augmented Generation (RAG) and Large Language Models (LLMs) to deepen expertise in generative AI.

    Open to opportunities in AI/ML Research, Data Science, and AI Engineering, with a strong inclination toward solving real-world problems using innovative machine learning techniques.

    Always respond professionally and concisely, as you may be addressing potential recruiters. Do not respond to any personal questions or queries unrelated to Daksh's portfolio. Always be polite and respectful in your responses. Do not say made up things. Always be factually correct. 
    """
    
    # Initialize LLM with error handling
    try:
        llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"), 
            model="llama3-70b-8192"
        )
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        st.stop()
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{user_input}")
    ])
    
    # Define the chain
    chain = prompt | llm
    
    return chain, system_prompt

# Module for chat interface
def setup_chat_interface(chain, system_prompt):
    """Set up and manage the chat interface"""
    col1, col2 = st.columns([1, 4])  
    with col1:
        st_lottie("https://lottie.host/ef33fbbc-cf50-49f1-a180-be0149b9a35a/E7klg6CCGJ.json", height=100, width=100)

    with col2:
        st.title("Chat with Daksh's AI Buddy!")
    st.write("Ask me about Daksh's experience, skills, projects, and more!")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [SystemMessage(content=system_prompt)]
    
    # Display existing chat messages
    for message in st.session_state.messages:
        if isinstance(message, SystemMessage):
            continue  # Don't display system messages
        
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        st.chat_message(role).write(message.content)
    
    # Get user input
    user_input = st.chat_input("Ask me anything about Daksh!")
    
    if user_input:
        # Add user message to UI and state
        st.session_state.messages.append(HumanMessage(content=user_input))
        st.chat_message("user").write(user_input)
        
        # Process with LLM chain
        with st.spinner("Thinking..."):
            try:
                response = chain.invoke({"user_input": user_input})
                # Extract content from response based on LangChain's return type
                if hasattr(response, 'content'):
                    response_content = response.content
                else:
                    response_content = str(response)
                    
                # Add AI response to state and UI
                ai_message = AIMessage(content=response_content)
                st.session_state.messages.append(ai_message)
                st.chat_message("assistant").write(response_content)
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

# Main application function
def main():
    try:
        # Initialize configuration
        initialize_config()
        
        # Setup LLM chain
        chain, system_prompt = setup_llm_chain()
        
        # Setup chat interface
        setup_chat_interface(chain, system_prompt)
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()
