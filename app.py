import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import json
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
from openai import OpenAI
from dotenv import load_dotenv

# Set page configuration
st.set_page_config(
    page_title="🏥 Health Assistant",
    layout="wide",
    page_icon="🧑‍⚕️",
    initial_sidebar_state="expanded"
)

# Initialize session state for history
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Initialize session state for chatbot history
if 'chatbot_history' not in st.session_state:
    st.session_state.chatbot_history = []

# --- CORRECTED PART ---
# Configure DeepSeek API using environment variables for better security
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
# --- END OF CORRECTION ---

# Custom CSS for black theme styling
st.markdown("""
<style>
    /* Main theme colors - Black based */
    .stApp {
        background-color: #000000;
        color: #ffffff;
    }
    
    .main-header {
        font-size: 3rem;
        color: #ffffff;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(255,255,255,0.1);
        background: linear-gradient(45deg, #1a1a1a, #2d2d2d);
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #333333;
    }
    
    .prediction-container {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 8px 32px rgba(255,255,255,0.1);
        border: 1px solid #333333;
    }
    
    .input-section {
        background: #1a1a1a;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #ffffff;
        border: 1px solid #333333;
    }
    
    .result-positive {
        background: linear-gradient(135deg, #cc0000, #990000);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
        border: 2px solid #ff3333;
    }
    
    .result-negative {
        background: linear-gradient(135deg, #006600, #004d00);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
        border: 2px solid #00cc00;
    }
    
    .info-card {
        background: #1a1a1a;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffffff;
        margin: 1rem 0;
        border: 1px solid #333333;
        color: #ffffff;
    }
    
    .history-card {
        background: #1a1a1a;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 1px solid #333333;
        color: #ffffff;
    }
    
    .metric-container {
        background: #1a1a1a;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #333333;
        color: #ffffff;
    }
    
    /* Override Streamlit's default colors */
    .stSelectbox > div > div {
        background-color: #1a1a1a;
        color: #ffffff;
        border: 1px solid #333333;
    }
    
    .stNumberInput > div > div > input {
        background-color: #1a1a1a;
        color: #ffffff;
        border: 1px solid #333333;
    }
    
    .stButton > button {
        background-color: #1a1a1a;
        color: #ffffff;
        border: 2px solid #ffffff;
        border-radius: 10px;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        background-color: #2d2d2d;
        border-color: #cccccc;
    }
    
    .stExpander {
        background-color: #1a1a1a;
        border: 1px solid #333333;
        border-radius: 10px;
    }
    
    .stDataFrame {
        background-color: #1a1a1a;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #0d0d0d;
    }
    
    .css-17eq0hr {
        background-color: #1a1a1a;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🏥 AI Health Assistant - Advanced Edition</h1>', unsafe_allow_html=True)
st.markdown("---")

# getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# loading the saved models
try:
    diabetes_model = pickle.load(open(f'{working_dir}/saved_models/diabetes_model.sav', 'rb'))
    heart_disease_model = pickle.load(open(f'{working_dir}/saved_models/heart_disease_model.sav', 'rb'))
    parkinsons_model = pickle.load(open(f'{working_dir}/saved_models/parkinsons_model.sav', 'rb'))
except FileNotFoundError:
    st.error("⚠️ Model files not found. Please ensure the saved_models directory exists with the required .sav files.")
    st.stop()

# Function to add prediction to history
def add_to_history(prediction_type, inputs, result, risk_level):
    history_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'type': prediction_type,
        'inputs': inputs,
        'result': result,
        'risk_level': risk_level
    }
    st.session_state.prediction_history.append(history_entry)

# Function to export history as JSON
def export_history_json():
    return json.dumps(st.session_state.prediction_history, indent=2)

# Function to export history as CSV
def export_history_csv():
    if not st.session_state.prediction_history:
        return None
    
    df_data = []
    for entry in st.session_state.prediction_history:
        row = {
            'Timestamp': entry['timestamp'],
            'Prediction Type': entry['type'],
            'Result': entry['result'],
            'Risk Level': entry['risk_level']
        }
        # Add input parameters as separate columns
        for key, value in entry['inputs'].items():
            row[f'Input_{key}'] = value
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    return df.to_csv(index=False)

# Function to create trend analysis charts
def create_trend_charts():
    if not st.session_state.prediction_history:
        st.info("No prediction history available for trend analysis.")
        return
    
    # Convert history to DataFrame for analysis
    df_data = []
    for entry in st.session_state.prediction_history:
        df_data.append({
            'timestamp': pd.to_datetime(entry['timestamp']),
            'type': entry['type'],
            'risk_level': entry['risk_level'],
            'result': 1 if entry['risk_level'] == 'High' else 0
        })
    
    df = pd.DataFrame(df_data)
    
    # Risk level distribution pie chart
    risk_counts = df['risk_level'].value_counts()
    fig_pie = px.pie(values=risk_counts.values, names=risk_counts.index, 
                     title="Overall Risk Level Distribution",
                     color_discrete_map={'High': '#cc0000', 'Low': '#006600'})
    fig_pie.update_layout(
        plot_bgcolor='#000000',
        paper_bgcolor='#1a1a1a',
        font_color='#ffffff'
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Prediction types distribution
    type_counts = df['type'].value_counts()
    fig_bar = px.bar(x=type_counts.index, y=type_counts.values,
                     title="Predictions by Disease Type",
                     color=type_counts.values,
                     color_continuous_scale='Viridis')
    fig_bar.update_layout(
        plot_bgcolor='#000000',
        paper_bgcolor='#1a1a1a',
        font_color='#ffffff',
        xaxis=dict(gridcolor='#333333'),
        yaxis=dict(gridcolor='#333333')
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Timeline of predictions
    if len(df) > 1:
        df_sorted = df.sort_values('timestamp')
        fig_timeline = px.scatter(df_sorted, x='timestamp', y='type', 
                                 color='risk_level', size_max=15,
                                 title="Prediction Timeline",
                                 color_discrete_map={'High': '#cc0000', 'Low': '#006600'})
        fig_timeline.update_layout(
            plot_bgcolor='#000000',
            paper_bgcolor='#1a1a1a',
            font_color='#ffffff',
            xaxis=dict(gridcolor='#333333'),
            yaxis=dict(gridcolor='#333333')
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

# sidebar for navigation
with st.sidebar:
    st.markdown("### 🩺 Navigation")
    selected = option_menu(
        'Health Assistant',
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction', 
         'Prediction History', 'Trend Analysis', 'Export Data', 'Health Chatbot'],
        icons=['🩸', '❤️', '🧠', '📋', '📊', '💾', '🤖'],
        menu_icon='hospital-fill',
        default_index=0,
        styles={
            "container": {
                "padding": "10px!important", 
                "background": "linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%)",
                "border-radius": "15px",
                "box-shadow": "0 8px 32px rgba(255,255,255,0.1)",
                "border": "2px solid #333333"
            },
            "icon": {
                "color": "#ffffff", 
                "font-size": "28px",
                "text-shadow": "2px 2px 4px rgba(0,0,0,0.3)"
            },
            "nav-link": {
                "font-size": "16px", 
                "text-align": "left", 
                "margin": "5px 0px",
                "padding": "12px 16px",
                "border-radius": "10px",
                "color": "#ffffff",
                "font-weight": "500",
                "transition": "all 0.3s ease",
                "--hover-color": "rgba(255,255,255,0.2)",
                "background": "rgba(255,255,255,0.1)",
                "backdrop-filter": "blur(10px)",
                "border": "1px solid #333333"
            },
            "nav-link-selected": {
                "background": "linear-gradient(135deg, #333333, #4d4d4d)",
                "color": "#ffffff",
                "font-weight": "bold",
                "box-shadow": "0 4px 15px rgba(255,255,255,0.2)",
                "transform": "translateY(-2px)",
                "border": "2px solid #ffffff"
            },
            "menu-title": {
                "color": "#ffffff",
                "font-weight": "bold",
                "text-align": "center",
                "font-size": "18px",
                "text-shadow": "2px 2px 4px rgba(0,0,0,0.3)",
                "margin-bottom": "20px"
            }
        }
    )
    
    # Add some info in sidebar
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info("""
    Enhanced AI health assistant with:
    - Disease Risk Prediction (Diabetes, Heart, Parkinson's)
    - Prediction History Tracking
    - Trend Analysis & Reporting
    - **Health Chatbot**
    """)
    
    # Quick stats
    if st.session_state.prediction_history:
        st.markdown("### 📊 Quick Stats")
        total_predictions = len(st.session_state.prediction_history)
        high_risk_count = sum(1 for entry in st.session_state.prediction_history if entry['risk_level'] == 'High')
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="metric-container"><h3>{total_predictions}</h3><p>Total Tests</p></div>', 
                       unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-container"><h3>{high_risk_count}</h3><p>High Risk</p></div>', 
                       unsafe_allow_html=True)

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.markdown("## 🩸 Diabetes Prediction using Machine Learning")
    
    # Information card
    with st.expander("📋 About Diabetes Prediction"):
        st.markdown("""
        **Diabetes** is a chronic condition that affects how your body processes blood sugar (glucose).
        This prediction model analyzes various health parameters to assess diabetes risk.
        
        **Key Parameters:**
        - 🤰 **Pregnancies**: Number of pregnancies
        - 🍭 **Glucose**: Blood glucose level
        - 💓 **Blood Pressure**: Systolic blood pressure
        - 📏 **BMI**: Body Mass Index
        """)

    #st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown("### 📝 Enter Your Health Parameters")
    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.number_input('🤰 Number of Pregnancies', min_value=0, max_value=20, value=0, help="Total number of pregnancies")
        SkinThickness = st.number_input('📏 Skin Thickness (mm)', min_value=0.0, max_value=100.0, value=20.0, help="Triceps skin fold thickness")
        DiabetesPedigreeFunction = st.number_input('🧬 Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.5, help="Family history factor")

    with col2:
        Glucose = st.number_input('🍭 Glucose Level (mg/dL)', min_value=0.0, max_value=300.0, value=120.0, help="Plasma glucose concentration")
        Insulin = st.number_input('💉 Insulin Level (μU/mL)', min_value=0.0, max_value=900.0, value=80.0, help="2-Hour serum insulin")

    with col3:
        BloodPressure = st.number_input('💓 Blood Pressure (mmHg)', min_value=0.0, max_value=200.0, value=80.0, help="Diastolic blood pressure")
        BMI = st.number_input('⚖️ BMI (kg/m²)', min_value=0.0, max_value=70.0, value=25.0, help="Body mass index")
        Age = st.number_input('🎂 Age (years)', min_value=1, max_value=120, value=30, help="Age in years")

    st.markdown('</div>', unsafe_allow_html=True)

    # code for Prediction
    diab_diagnosis = ''

    # creating a button for Prediction
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button('🔬 Run Diabetes Test', type="primary", use_container_width=True):
            try:
                user_input = [float(Pregnancies), float(Glucose), float(BloodPressure), 
                             float(SkinThickness), float(Insulin), float(BMI), 
                             float(DiabetesPedigreeFunction), float(Age)]

                diab_prediction = diabetes_model.predict([user_input])
                
                # Store inputs for history
                inputs = {
                    'Pregnancies': Pregnancies, 'Glucose': Glucose, 'BloodPressure': BloodPressure,
                    'SkinThickness': SkinThickness, 'Insulin': Insulin, 'BMI': BMI,
                    'DiabetesPedigreeFunction': DiabetesPedigreeFunction, 'Age': Age
                }

                if diab_prediction[0] == 1:
                    st.markdown('<div class="result-positive">⚠️ High Risk: The model suggests increased diabetes risk</div>', unsafe_allow_html=True)
                    st.warning("⚠️ This prediction suggests a higher risk of diabetes. Please consult with a healthcare professional for proper evaluation and testing.")
                    add_to_history('Diabetes Prediction', inputs, 'Positive', 'High')
                else:
                    st.markdown('<div class="result-negative">✅ Low Risk: The model suggests lower diabetes risk</div>', unsafe_allow_html=True)
                    st.success("✅ This prediction suggests a lower risk of diabetes. Continue maintaining a healthy lifestyle!")
                    add_to_history('Diabetes Prediction', inputs, 'Negative', 'Low')
    
            except ValueError:
                st.error("❌ Please ensure all fields are filled with valid numbers.")

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    st.markdown("## ❤️ Heart Disease Prediction using Machine Learning")
    
    # Information card
    with st.expander("📋 About Heart Disease Prediction"):
        st.markdown("""
        **Heart Disease** refers to various conditions that affect heart function.
        This model analyzes cardiovascular parameters to assess heart disease risk.
        
        **Key Parameters:**
        - 👥 **Demographics**: Age, Sex
        - 💓 **Cardiovascular**: Blood pressure, cholesterol, heart rate
        - 🔬 **Clinical Tests**: ECG results, stress tests
        """)

    # st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown("### 📝 Enter Your Cardiovascular Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('🎂 Age (years)', min_value=1, max_value=120, value=50)
        trestbps = st.number_input('💓 Resting Blood Pressure (mmHg)', min_value=50.0, max_value=250.0, value=120.0)
        restecg = st.selectbox('📊 Resting ECG Results', options=[0, 1, 2], format_func=lambda x: {0: "Normal", 1: "ST-T Abnormality", 2: "LV Hypertrophy"}[x])
        oldpeak = st.number_input('📈 ST Depression', min_value=0.0, max_value=10.0, value=1.0, step=0.1)

    with col2:
        sex = st.selectbox('👤 Sex', options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        chol = st.number_input('🧪 Serum Cholesterol (mg/dl)', min_value=100.0, max_value=600.0, value=200.0)
        thalach = st.number_input('💨 Max Heart Rate Achieved', min_value=50.0, max_value=250.0, value=150.0)
        slope = st.selectbox('📊 ST Segment Slope', options=[0, 1, 2], format_func=lambda x: {0: "Upsloping", 1: "Flat", 2: "Downsloping"}[x])

    with col3:
        cp = st.selectbox('🫀 Chest Pain Type', options=[0, 1, 2, 3], format_func=lambda x: {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-anginal", 3: "Asymptomatic"}[x])
        fbs = st.selectbox('🍭 Fasting Blood Sugar > 120 mg/dl', options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        exang = st.selectbox('🏃 Exercise Induced Angina', options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        ca = st.selectbox('🔬 Major Vessels (0-3)', options=[0, 1, 2, 3])

    thal = st.selectbox('🫀 Thalassemia', options=[0, 1, 2], format_func=lambda x: {0: "Normal", 1: "Fixed Defect", 2: "Reversible Defect"}[x])

    st.markdown('</div>', unsafe_allow_html=True)

    # code for Prediction
    heart_diagnosis = ''

    # creating a button for Prediction
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button('🔬 Run Heart Disease Test', type="primary", use_container_width=True):
            try:
                user_input = [float(age), float(sex), float(cp), float(trestbps), float(chol), 
                             float(fbs), float(restecg), float(thalach), float(exang), 
                             float(oldpeak), float(slope), float(ca), float(thal)]

                heart_prediction = heart_disease_model.predict([user_input])
                
                # Store inputs for history
                inputs = {
                    'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
                    'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
                    'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
                }

                if heart_prediction[0] == 1:
                    st.markdown('<div class="result-positive">⚠️ High Risk: The model suggests increased heart disease risk</div>', unsafe_allow_html=True)
                    st.warning("⚠️ This prediction suggests a higher risk of heart disease. Please consult with a cardiologist for proper evaluation.")
                    add_to_history('Heart Disease Prediction', inputs, 'Positive', 'High')
                else:
                    st.markdown('<div class="result-negative">✅ Low Risk: The model suggests lower heart disease risk</div>', unsafe_allow_html=True)
                    st.success("✅ This prediction suggests a lower risk of heart disease. Keep up the healthy lifestyle!")
                    add_to_history('Heart Disease Prediction', inputs, 'Negative', 'Low')
                    
            except ValueError:
                st.error("❌ Please ensure all fields are filled with valid values.")

# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":
    st.markdown("## 🧠 Parkinson's Disease Prediction using Machine Learning")
    
    # Information card
    with st.expander("📋 About Parkinson's Disease Prediction"):
        st.markdown("""
        **Parkinson's Disease** is a neurodegenerative disorder affecting movement.
        This model analyzes voice measurements to assess Parkinson's disease risk.
        
        **Voice Analysis Parameters:**
        - 🎵 **Frequency**: Fundamental frequency variations
        - 📊 **Jitter**: Frequency variation measures
        - 🌊 **Shimmer**: Amplitude variation measures
        - 📈 **Harmonic Ratios**: Voice quality indicators
        """)

    # st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown("### 📝 Enter Voice Analysis Parameters")
    
    st.info("🎤 These parameters are typically obtained from voice recording analysis. Consult with a healthcare professional for proper voice analysis.")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.number_input('🎵 MDVP:Fo(Hz)', min_value=50.0, max_value=300.0, value=150.0, help="Average vocal fundamental frequency")
        RAP = st.number_input('📊 MDVP:RAP', min_value=0.0, max_value=1.0, value=0.01, step=0.001, format="%.4f")
        Shimmer = st.number_input('🌊 MDVP:Shimmer', min_value=0.0, max_value=1.0, value=0.03, step=0.001, format="%.4f")
        APQ = st.number_input('📈 MDVP:APQ', min_value=0.0, max_value=1.0, value=0.02, step=0.001, format="%.4f")
        RPDE = st.number_input('🔢 RPDE', min_value=0.0, max_value=1.0, value=0.5, step=0.01)

    with col2:
        fhi = st.number_input('🎵 MDVP:Fhi(Hz)', min_value=50.0, max_value=500.0, value=200.0)
        PPQ = st.number_input('📊 MDVP:PPQ', min_value=0.0, max_value=1.0, value=0.01, step=0.001, format="%.4f")
        Shimmer_dB = st.number_input('🌊 MDVP:Shimmer(dB)', min_value=0.0, max_value=2.0, value=0.3, step=0.01)
        DDA = st.number_input('📈 Shimmer:DDA', min_value=0.0, max_value=1.0, value=0.05, step=0.001, format="%.4f")
        DFA = st.number_input('🔢 DFA', min_value=0.0, max_value=1.0, value=0.7, step=0.01)

    with col3:
        flo = st.number_input('🎵 MDVP:Flo(Hz)', min_value=50.0, max_value=300.0, value=100.0)
        DDP = st.number_input('📊 Jitter:DDP', min_value=0.0, max_value=1.0, value=0.03, step=0.001, format="%.4f")
        APQ3 = st.number_input('🌊 Shimmer:APQ3', min_value=0.0, max_value=1.0, value=0.02, step=0.001, format="%.4f")
        NHR = st.number_input('📈 NHR', min_value=0.0, max_value=1.0, value=0.02, step=0.001, format="%.4f")
        spread1 = st.number_input('📊 spread1', min_value=-10.0, max_value=0.0, value=-5.0, step=0.1)

    with col4:
        Jitter_percent = st.number_input('📊 MDVP:Jitter(%)', min_value=0.0, max_value=10.0, value=0.5, step=0.01)
        APQ5 = st.number_input('🌊 Shimmer:APQ5', min_value=0.0, max_value=1.0, value=0.02, step=0.001, format="%.4f")
        HNR = st.number_input('📈 HNR', min_value=0.0, max_value=50.0, value=20.0, step=0.1)
        spread2 = st.number_input('📊 spread2', min_value=0.0, max_value=1.0, value=0.2, step=0.01)

    with col5:
        Jitter_Abs = st.number_input('📊 MDVP:Jitter(Abs)', min_value=0.0, max_value=1.0, value=0.0001, step=0.0001, format="%.6f")
        D2 = st.number_input('🔢 D2', min_value=0.0, max_value=5.0, value=2.0, step=0.1)
        PPE = st.number_input('🔢 PPE', min_value=0.0, max_value=1.0, value=0.2, step=0.01)

    st.markdown('</div>', unsafe_allow_html=True)

    # creating a button for Prediction    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔬 Run Parkinson's Test", type="primary", use_container_width=True):
            try:
                user_input = [float(fo), float(fhi), float(flo), float(Jitter_percent), float(Jitter_Abs),
                              float(RAP), float(PPQ), float(DDP), float(Shimmer), float(Shimmer_dB), 
                              float(APQ3), float(APQ5), float(APQ), float(DDA), float(NHR), 
                              float(HNR), float(RPDE), float(DFA), float(spread1), float(spread2), 
                              float(D2), float(PPE)]

                parkinsons_prediction = parkinsons_model.predict([user_input])
                
                # Store inputs for history
                inputs = {
                    'fo': fo, 'fhi': fhi, 'flo': flo, 'Jitter_percent': Jitter_percent, 'Jitter_Abs': Jitter_Abs,
                    'RAP': RAP, 'PPQ': PPQ, 'DDP': DDP, 'Shimmer': Shimmer, 'Shimmer_dB': Shimmer_dB,
                    'APQ3': APQ3, 'APQ5': APQ5, 'APQ': APQ, 'DDA': DDA, 'NHR': NHR, 'HNR': HNR,
                    'RPDE': RPDE, 'DFA': DFA, 'spread1': spread1, 'spread2': spread2, 'D2': D2, 'PPE': PPE
                }

                if parkinsons_prediction[0] == 1:
                    st.markdown('<div class="result-positive">⚠️ High Risk: The model suggests increased Parkinson\'s disease risk</div>', unsafe_allow_html=True)
                    st.warning("⚠️ This prediction suggests a higher risk of Parkinson's disease. Please consult with a neurologist for proper evaluation.")
                    add_to_history('Parkinsons Prediction', inputs, 'Positive', 'High')
                else:
                    st.markdown('<div class="result-negative">✅ Low Risk: The model suggests lower Parkinson\'s disease risk</div>', unsafe_allow_html=True)
                    st.success("✅ This prediction suggests a lower risk of Parkinson's disease. Regular health monitoring is still recommended.")
                    add_to_history('Parkinsons Prediction', inputs, 'Negative', 'Low')
                    
            except ValueError:
                st.error("❌ Please ensure all fields are filled with valid numbers.")

# Prediction History Page
if selected == 'Prediction History':
    st.markdown("## 📋 Prediction History")
    
    if not st.session_state.prediction_history:
        st.info("No prediction history available. Make some predictions first!")
    else:
        st.markdown(f"### Total Predictions: {len(st.session_state.prediction_history)}")
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        with col1:
            prediction_types = ['All'] + list(set(entry['type'] for entry in st.session_state.prediction_history))
            selected_type = st.selectbox("Filter by Prediction Type", prediction_types)
        
        with col2:
            risk_levels = ['All', 'High', 'Low']
            selected_risk = st.selectbox("Filter by Risk Level", risk_levels)
        
        with col3:
            if st.button("🗑️ Clear History", type="secondary"):
                st.session_state.prediction_history = []
                st.rerun()
        
        # Filter history
        filtered_history = st.session_state.prediction_history
        if selected_type != 'All':
            filtered_history = [entry for entry in filtered_history if entry['type'] == selected_type]
        if selected_risk != 'All':
            filtered_history = [entry for entry in filtered_history if entry['risk_level'] == selected_risk]
        
        # Display history
        for i, entry in enumerate(reversed(filtered_history)):
            with st.container():
                st.markdown(f'<div class="history-card">', unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
                with col1:
                    st.markdown(f"**📅 {entry['timestamp']}**")
                with col2:
                    st.markdown(f"**🔬 {entry['type']}**")
                with col3:
                    risk_color = "#cc0000" if entry['risk_level'] == 'High' else "#006600"
                    st.markdown(f"<span style='color: {risk_color}; font-weight: bold;'>🚨 {entry['risk_level']} Risk</span>", unsafe_allow_html=True)
                with col4:
                    st.markdown(f"**📊 {entry['result']}**")
                
                # Show input parameters in expandable section
                with st.expander(f"View Input Parameters - Test #{len(filtered_history) - i}"):
                    input_cols = st.columns(3)
                    for idx, (key, value) in enumerate(entry['inputs'].items()):
                        with input_cols[idx % 3]:
                            st.write(f"**{key}:** {value}")
                
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown("---")

# Trend Analysis Page
if selected == 'Trend Analysis':
    st.markdown("## 📊 Trend Analysis & Progress Monitoring")
    
    if not st.session_state.prediction_history:
        st.info("No prediction history available for trend analysis. Make some predictions first!")
    else:
        # Overview metrics
        st.markdown("### 📈 Overview Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        total_predictions = len(st.session_state.prediction_history)
        high_risk_predictions = sum(1 for entry in st.session_state.prediction_history if entry['risk_level'] == 'High')
        diabetes_tests = sum(1 for entry in st.session_state.prediction_history if 'Diabetes' in entry['type'])
        recent_tests = sum(1 for entry in st.session_state.prediction_history 
                          if datetime.now() - datetime.strptime(entry['timestamp'], '%Y-%m-%d %H:%M:%S') <= timedelta(days=7))
        
        with col1:
            st.markdown(f'<div class="metric-container"><h2>{total_predictions}</h2><p>Total Predictions</p></div>', 
                       unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-container"><h2>{high_risk_predictions}</h2><p>High Risk Results</p></div>', 
                       unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-container"><h2>{diabetes_tests}</h2><p>Diabetes Tests</p></div>', 
                       unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-container"><h2>{recent_tests}</h2><p>Tests This Week</p></div>', 
                       unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Risk percentage
        risk_percentage = (high_risk_predictions / total_predictions) * 100 if total_predictions > 0 else 0
        st.markdown(f"### 🎯 Overall Risk Assessment: **{risk_percentage:.1f}%** High Risk")
        
        # Progress bar for risk level
        progress_color = "#cc0000" if risk_percentage > 50 else "#ff9900" if risk_percentage > 25 else "#006600"
        st.markdown(f'''
        <div style="background-color: #1a1a1a; border-radius: 10px; padding: 10px; border: 1px solid #333333;">
            <div style="background-color: {progress_color}; height: 20px; border-radius: 10px; width: {risk_percentage}%;"></div>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Trend charts
        st.markdown("### 📊 Visual Trend Analysis")
        create_trend_charts()
        
        # Recent activity
        st.markdown("### 🕒 Recent Activity (Last 7 Days)")
        recent_entries = [entry for entry in st.session_state.prediction_history 
                         if datetime.now() - datetime.strptime(entry['timestamp'], '%Y-%m-%d %H:%M:%S') <= timedelta(days=7)]
        
        if recent_entries:
            for entry in reversed(recent_entries[-5:]):  # Show last 5 recent entries
                risk_emoji = "🔴" if entry['risk_level'] == 'High' else "🟢"
                st.markdown(f"{risk_emoji} **{entry['type']}** - {entry['risk_level']} Risk - *{entry['timestamp']}*")
        else:
            st.info("No recent activity in the last 7 days.")

# Export Data Page
if selected == 'Export Data':
    st.markdown("## 💾 Export Data & Reports")
    
    if not st.session_state.prediction_history:
        st.info("No prediction history available for export. Make some predictions first!")
    else:
        st.markdown("### 📊 Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown("#### 📄 JSON Export")
            st.markdown("Export complete prediction history with all parameters in JSON format.")
            
            json_data = export_history_json()
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name=f"health_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown("#### 📊 CSV Export")
            st.markdown("Export prediction history in CSV format for spreadsheet analysis.")
            
            csv_data = export_history_csv()
            if csv_data:
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name=f"health_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Generate comprehensive report
        st.markdown("### 📋 Comprehensive Health Report")
        
        if st.button("📊 Generate Report", type="primary", use_container_width=True):
            # Start report string
            report = f"""# 🏥 Comprehensive Health Report
**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Summary Statistics
- **Total Predictions:** {len(st.session_state.prediction_history)}
- **High Risk Results:** {sum(1 for entry in st.session_state.prediction_history if entry['risk_level'] == 'High')}
- **Low Risk Results:** {sum(1 for entry in st.session_state.prediction_history if entry['risk_level'] == 'Low')}
- **Risk Percentage:** {(sum(1 for entry in st.session_state.prediction_history if entry['risk_level'] == 'High') / len(st.session_state.prediction_history) * 100):.1f}%

## 🔬 Test Breakdown
"""
        
            # Count by prediction type
            prediction_counts = {}
            for entry in st.session_state.prediction_history:
                pred_type = entry['type']
                if pred_type in prediction_counts:
                    prediction_counts[pred_type]['total'] += 1
                    if entry['risk_level'] == 'High':
                        prediction_counts[pred_type]['high_risk'] += 1
                else:
                    prediction_counts[pred_type] = {'total': 1, 'high_risk': 1 if entry['risk_level'] == 'High' else 0}
            
            for pred_type, counts in prediction_counts.items():
                risk_rate = (counts['high_risk'] / counts['total'] * 100) if counts['total'] > 0 else 0
                report += f"- **{pred_type}:** {counts['total']} tests, {counts['high_risk']} high risk ({risk_rate:.1f}%)\n"

            report += f"""\n
## 📈 Recent Activity
Last 5 predictions:
"""
        
            # Add recent predictions
            recent_predictions = sorted(st.session_state.prediction_history,
                                        key=lambda x: x['timestamp'], reverse=True)[:5]
            for entry in recent_predictions:
                risk_indicator = "🔴" if entry['risk_level'] == 'High' else "🟢"
                report += f"- {risk_indicator} {entry['timestamp']} - {entry['type']} - {entry['risk_level']} Risk\n"
            
            report += f"""\n
## ⚠️ Important Disclaimer
This AI-powered health assistant is for informational purposes only and should not replace professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for proper medical evaluation and care.

## 📞 Recommendations
"""
        
            # Add personalized recommendations
            high_risk_count = sum(1 for entry in st.session_state.prediction_history if entry['risk_level'] == 'High')
            total_count = len(st.session_state.prediction_history)
            
            if high_risk_count > total_count * 0.5:
                report += "- 🚨 **High Priority:** Consider scheduling appointments with relevant specialists\n"
                report += "- 📋 **Medical Records:** Bring this report to your healthcare provider\n"
                report += "- 🔄 **Regular Monitoring:** Consider more frequent health checkups\n"
            elif high_risk_count > 0:
                report += "- ⚠️ **Moderate Priority:** Discuss results with your primary care physician\n"
                report += "- 🌱 **Lifestyle:** Focus on healthy diet, exercise, and stress management\n"
                report += "- 📊 **Monitoring:** Continue regular health assessments\n"
            else:
                report += "- ✅ **Maintenance:** Continue your current healthy lifestyle\n"
                report += "- 🔄 **Prevention:** Regular health screenings as recommended by age\n"
                report += "- 📈 **Wellness:** Consider preventive health measures\n"
            
            # Display the report
            st.markdown(report)
            
            # Download report
            st.download_button(
                label="📥 Download Report",
                data=report,
                file_name=f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True
            )

# Health Chatbot page
if selected == 'Health Chatbot':
    st.markdown("## 🤖 Health Chatbot - Powered by DeepSeek")
    st.info("💡 Ask me anything about symptoms, diseases, or healthy living. I am an AI and my advice is not a substitute for a real doctor.")

    # Check if the API key is configured properly.
    if not DEEPSEEK_API_KEY:
        st.error("⚠️ DEEPSEEK_API_KEY environment variable not set. Please create a `.env` file and add your key.")
        st.stop()

    # Initialize the client in session_state if it doesn't exist.
    if "deepseek_client" not in st.session_state:
        st.session_state.deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
        
    if "chatbot_history" not in st.session_state:
        # We now store messages with roles for better context management
        st.session_state.chatbot_history = []

    # Display all the messages stored in the session state
    for message in st.session_state.chatbot_history:
        # Use the 'role' to determine the avatar and chat message alignment
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if user_prompt := st.chat_input("Ask a health-related question..."):
        
        # Add user's message to the history and display it
        st.session_state.chatbot_history.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Generate and display the AI's response
        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):
                try:
                    # Prepare the message history for the API call
                    messages_for_api = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.chatbot_history
                    ]
                    
                    # Get the AI response from DeepSeek
                    response = st.session_state.deepseek_client.chat.completions.create(
                        model="deepseek-chat",
                        messages=messages_for_api,
                    )
                    bot_reply = response.choices[0].message.content
                    
                    # Add AI's response to the history
                    st.session_state.chatbot_history.append({"role": "assistant", "content": bot_reply})
                    
                    # Display the AI response
                    st.markdown(bot_reply)

                except Exception as e:
                    # Updated error message to be more specific
                    error_message = f"❌ An error occurred: {e}. Please check your API key's validity and ensure your DeepSeek account has sufficient balance."
                    st.error(error_message)
                    # Optionally, add the error to history so it's visible
                    st.session_state.chatbot_history.append({"role": "assistant", "content": error_message})