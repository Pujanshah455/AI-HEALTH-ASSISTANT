# 🏥 AI Health Assistant - Advanced Edition

A comprehensive machine learning-powered health prediction application built with Streamlit. This application provides risk assessment for three major health conditions: Diabetes, Heart Disease, and Parkinson's Disease with 82 % Accuracy, along with advanced features like prediction history tracking, trend analysis, and data export capabilities.

## 🌟 Features

### Core Predictions
- **🩸 Diabetes Prediction**: Risk assessment based on glucose levels, BMI, blood pressure, and other health parameters
- **❤️ Heart Disease Prediction**: Cardiovascular risk analysis using ECG results, cholesterol levels, and clinical data
- **🧠 Parkinson's Disease Prediction**: Voice analysis-based prediction using frequency and amplitude measurements

### Advanced Features
- **📋 Prediction History**: Track all your health assessments with timestamps
- **📊 Trend Analysis**: Visual charts and progress monitoring
- **💾 Data Export**: Export predictions in JSON/CSV formats
- **📄 Comprehensive Reports**: Generate detailed health reports with recommendations
- **🎨 Modern UI**: Dark theme with professional styling and responsive design

### Live Demo 
- 🔗 https://ai-health-assistant-by-aksh-patel.streamlit.app/

## 🗂️ Project Structure

```
ai-health-assistant/
├── main.py                 
├── saved_models/           
│   ├── diabetes_model.sav
│   ├── heart_disease_model.sav
│   └── parkinsons_model.sav
├── requirements.txt       
├── README.md                          
```
 
## 🚀 Quick Start

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/akshpatel26/AI-Health-Assistant.git
   cd ai-health-assistant
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up model files**
   - Create a `saved_models` directory in the project root
   - Place your trained model files:
     - `diabetes_model.sav`
     - `heart_disease_model.sav`
     - `parkinsons_model.sav`

4. **Run the application**
   ```bash
   streamlit run main.py
   ```

5. **Open your browser**
   - Navigate to `http://localhost:8501`

## 📦 Dependencies

```txt
streamlit>=1.28.0
streamlit-option-menu>=0.3.6
pandas>=1.5.0
plotly>=5.15.0
scikit-learn>=1.3.0
pickle-mixin>=1.0.2
```


## 🎯 Usage Guide

### ➤  Making Predictions
1. **Select a prediction type** from the sidebar navigation
2. **Enter the required parameters** in the input fields
3. **Click the prediction button** to get results
4. **View the risk assessment** and recommendations

### ➤ Viewing History
1. Navigate to "Prediction History"
2. Filter by prediction type or risk level
3. Expand entries to view detailed input parameters
4. Clear history if needed

### ➤ Trend Analysis
1. Go to "Trend Analysis" section
2. View overview metrics and risk percentages
3. Analyze visual charts showing prediction patterns
4. Monitor recent activity and progress

### ➤  Exporting Data
1. Visit the "Export Data" section
2. Choose between JSON or CSV format
3. Generate comprehensive health reports
4. Download files for external analysis

"# Health-Assistant-main" 
