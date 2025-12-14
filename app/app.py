import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

@st.cache_resource
def load_model():
    """Cache model loading"""
    model_paths = [
        "models/churn_model.pkl",
        "../models/churn_model.pkl",
        "./models/churn_model.pkl"
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            try:
                return joblib.load(path)
            except Exception as e:
                st.error(f"Error loading model from {path}: {e}")
    
    st.error("Model file not found. Please check the models/ directory.")
    st.stop()

@st.cache_data
def load_training_structure():
    """Load and understand the training data structure"""
    data_paths = [
        "data/churn_cleaned.csv",
        "../data/churn_cleaned.csv",
        "./data/churn_cleaned.csv"
    ]
    
    for path in data_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, nrows=1000)
                
                # Identify which columns need one-hot encoding
                categorical_cols = ['InternetService', 'Contract', 'PaymentMethod']
                
                # Create one-hot encoded version to see what the model expects
                categorical_data = df[categorical_cols]
                encoded = pd.get_dummies(categorical_data, drop_first=False)
                
                # All feature columns = numeric columns + encoded columns
                numeric_cols = [col for col in df.columns if col not in categorical_cols + ['Churn']]
                all_feature_cols = numeric_cols + encoded.columns.tolist()
                
                return all_feature_cols
                
            except Exception as e:
                st.error(f"Error loading data from {path}: {e}")
    
    st.error("Training data file not found. Please check the data/ directory.")
    st.stop()

# Load model and structure
model = load_model()
all_feature_cols = load_training_structure()

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

st.markdown(
    """
    <style>
    :root{
        --card-bg: #141414;
        --card-secondary: #1e1e1e;
        --accent: #00bfff;
        --muted: #cfcfcf;
        --text: #f0f0f0;
    }

    /* Body */
    .stApp {
        background-color: #0f0f0f;
        color: var(--text);
    }

    /* Centered header with floating animation */
    .header-wrap {
        text-align: center;
        margin-bottom: 8px;
        animation: floatTitle 6s ease-in-out infinite;
    }
    @keyframes floatTitle {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-6px); }
        100% { transform: translateY(0px); }
    }

    /* Prediction bar container */
    .pred-wrap {
        margin-top: 12px;
        margin-bottom: 12px;
    }
    .pred-bar {
        height: 54px;
        border-radius: 12px;
        background: linear-gradient(90deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
        border: 1px solid rgba(255,255,255,0.04);
        position: relative;
        overflow: hidden;
        box-shadow: 0 8px 30px rgba(0,0,0,0.6);
    }
    .pred-fill {
        height:100%;
        width: 0%;
        border-radius: 12px;
        display: flex;
        align-items:center;
        justify-content:center;
        color: #0b0b0b;
        font-weight:700;
        transition: width 1.2s cubic-bezier(.2,.9,.2,1);
    }

    /* glowing pulsing shadow */
    .glow {
        animation: pulseGlow 1.6s infinite;
    }
    @keyframes pulseGlow {
        0% { box-shadow: 0 0 8px rgba(0,0,0,0.0); }
        50% { box-shadow: 0 0 30px currentColor; }
        100% { box-shadow: 0 0 8px rgba(0,0,0,0.0); }
    }

    /* Risk label */
    .risk-label {
        display:inline-block;
        padding:6px 12px;
        border-radius:999px;
        font-weight:700;
        color:#0f0f0f;
        margin-top:8px;
    }

    /* small text muted */
    .muted { color: var(--muted); font-size:13px; }

    /* responsive spacing */
    .spacer { height: 8px; }
    
    /* Fix for plot background */
    .stPlot { background-color: transparent !important; }
    
    /* Better button styling */
    .stButton > button {
        background: linear-gradient(90deg, var(--accent), #0099cc);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 191, 255, 0.4);
    }
    
    /* Section titles without boxes */
    .section-title {
        color: var(--accent);
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 2px solid rgba(0, 191, 255, 0.2);
    }
    
    /* Input groups styling */
    .input-group {
        background: rgba(20, 20, 20, 0.7);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    
    /* Insights card styling */
    .insights-card {
        background: rgba(20, 20, 20, 0.7);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="header-wrap">
        <h1 style="margin:0;color:var(--accent);">📊 Customer Churn Predictor</h1>
        <p style="margin-top:6px;color:var(--muted);font-size:15px;">
            Predict whether a customer will churn.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

def three_inputs():
    c1, c2, c3 = st.columns([1,1,1])
    return c1, c2, c3

# Section 1: Customer Info
st.markdown('<div class="section-title">👤 Customer Info</div>', unsafe_allow_html=True)

c1, c2, c3 = three_inputs()
with c1:
    gender_display = st.selectbox("Gender", ["Male", "Female"], help="Customer gender")
    gender = 1 if gender_display == "Male" else 0

with c2:
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", help="1 if senior citizen")

with c3:
    partner_display = st.selectbox("Partner", ["Yes", "No"], help="Has partner?")
    Partner = 1 if partner_display == "Yes" else 0

c1, c2, c3 = three_inputs()
with c1:
    dependents_display = st.selectbox("Dependents", ["Yes", "No"], help="Has dependents?")
    Dependents = 1 if dependents_display == "Yes" else 0

with c2:
    tenure = st.number_input("Tenure (months)", 0, 72, 12, help="Months with company")

with c3:
    phone_display = st.selectbox("Phone Service", ["Yes", "No"], help="Has phone service")
    PhoneService = 1 if phone_display == "Yes" else 0

st.markdown('</div>', unsafe_allow_html=True)

# Section 2: Services
st.markdown('<div class="section-title">💻 Services</div>', unsafe_allow_html=True)

c1, c2, c3 = three_inputs()
with c1:
    multiple_display = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"], help="Multiple phone lines")
    MultipleLines = 1 if multiple_display == "Yes" else 0

with c2:
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], help="Type of internet service")

with c3:
    security_display = st.selectbox("Online Security", ["Yes", "No", "No internet service"], help="Online security subscribed")
    OnlineSecurity = 1 if security_display == "Yes" else 0

c1, c2, c3 = three_inputs()
with c1:
    backup_display = st.selectbox("Online Backup", ["Yes", "No", "No internet service"], help="Online backup service")
    OnlineBackup = 1 if backup_display == "Yes" else 0

with c2:
    device_display = st.selectbox("Device Protection", ["Yes", "No", "No internet service"], help="Device protection")
    DeviceProtection = 1 if device_display == "Yes" else 0

with c3:
    tech_display = st.selectbox("Tech Support", ["Yes", "No", "No internet service"], help="Technical support")
    TechSupport = 1 if tech_display == "Yes" else 0

c1, c2, c3 = three_inputs()
with c1:
    tv_display = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"], help="TV streaming")
    StreamingTV = 1 if tv_display == "Yes" else 0

with c2:
    movies_display = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"], help="Movie streaming")
    StreamingMovies = 1 if movies_display == "Yes" else 0

with c3:
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], help="Contract type")

st.markdown('</div>', unsafe_allow_html=True)

# Section 3: Billing & Payment
st.markdown('<div class="section-title">💳 Billing & Payment</div>', unsafe_allow_html=True)

c1, c2, c3 = three_inputs()
with c1:
    paperless_display = st.selectbox("Paperless Billing", ["Yes", "No"], help="Paperless billing")
    PaperlessBilling = 1 if paperless_display == "Yes" else 0

with c2:
    PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], help="Payment method")

with c3:
    MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0, step=1.0, help="Monthly bill")

c1, c2, c3 = three_inputs()
with c1:
    TotalCharges = st.number_input("Total Charges", 0.0, 10000.0, 200.0, step=1.0, help="Total bill to date")
with c2:
    pass  
with c3:
    pass  

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

def create_one_hot_encoded_input(input_dict, all_feature_cols):
    """
    Create properly one-hot encoded input matching model expectations
    """
    # Start with numeric columns
    numeric_data = {
        "gender": input_dict["gender"],
        "SeniorCitizen": input_dict["SeniorCitizen"],
        "Partner": input_dict["Partner"],
        "Dependents": input_dict["Dependents"],
        "tenure": input_dict["tenure"],
        "PhoneService": input_dict["PhoneService"],
        "MultipleLines": input_dict["MultipleLines"],
        "OnlineSecurity": input_dict["OnlineSecurity"],
        "OnlineBackup": input_dict["OnlineBackup"],
        "DeviceProtection": input_dict["DeviceProtection"],
        "TechSupport": input_dict["TechSupport"],
        "StreamingTV": input_dict["StreamingTV"],
        "StreamingMovies": input_dict["StreamingMovies"],
        "PaperlessBilling": input_dict["PaperlessBilling"],
        "MonthlyCharges": input_dict["MonthlyCharges"],
        "TotalCharges": input_dict["TotalCharges"]
    }
    
    # Create DataFrame for numeric data
    df = pd.DataFrame([numeric_data])
    
    # One-hot encode categorical columns
    categorical_data = {
        "InternetService": [input_dict["InternetService"]],
        "Contract": [input_dict["Contract"]],
        "PaymentMethod": [input_dict["PaymentMethod"]]
    }
    
    cat_df = pd.DataFrame(categorical_data)
    encoded_cats = pd.get_dummies(cat_df, drop_first=False)
    
    # Combine numeric and encoded categorical data
    final_df = pd.concat([df, encoded_cats], axis=1)
    
    # Ensure all expected columns are present
    for col in all_feature_cols:
        if col not in final_df.columns:
            final_df[col] = 0
    
    # Reorder columns to match expected order
    final_df = final_df[all_feature_cols]
    
    return final_df

if st.button("🔮 Predict Churn", use_container_width=True, type="primary"):
    # Collect all inputs
    input_dict = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }
    
    try:
        with st.spinner("Processing..."):
            # Create properly encoded input
            input_processed = create_one_hot_encoded_input(input_dict, all_feature_cols)
            
            with st.spinner("Making prediction..."):
                # Get probability if available
                if hasattr(model, "predict_proba"):
                    prob = float(model.predict_proba(input_processed)[0][1])
                    pred = 1 if prob > 0.35 else 0
                else:
                    pred = model.predict(input_processed)[0]
                    prob = 0.8 if pred == 1 else 0.2
        
        # Visualization
        if prob < 0.4:
            color = "#FFD700"   
            risk_text = "Low Risk"
            risk_bg = "#FFD700"
        elif prob < 0.7:
            color = "#FFA500"   
            risk_text = "Medium Risk"
            risk_bg = "#FFA500"
        else:
            color = "#FF4500"   
            risk_text = "High Risk"
            risk_bg = "#FF4500"
        
        width_pct = min(int(prob * 100), 100)
        prediction_text = "CHURN" if pred == 1 else "NO CHURN"
        
        st.markdown(
            f"""
            <div class="pred-wrap">
                <div class="pred-bar">
                    <div class="pred-fill glow" style="
                        width:{width_pct}%;
                        background: linear-gradient(90deg, {color} 0%, rgba(255,255,255,0.15) 100%);
                        color: {color};
                    ">
                        <div style="width:100%; text-align:center; color:#0b0b0b; font-weight:800;">
                            {width_pct}% &nbsp; {prediction_text}
                        </div>
                    </div>
                </div>
            </div>
            <div style="height:8px;"></div>
            <div style="display:flex; gap:12px; align-items:center;">
                <div class="risk-label" style="background:{risk_bg};">{risk_text}</div>
                <div class="muted">Probability: {prob:.3f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        st.markdown('<div class="section-title">💡 Insights & Recommendations</div>', unsafe_allow_html=True)
        st.markdown('<div class="insights-card">', unsafe_allow_html=True)
        
        insights = []
        if Contract == "Month-to-month":
            insights.append("**High Risk Factor**: Month-to-month contract (most flexible, highest churn)")
        if tenure < 6:
            insights.append(f"**Risk Factor**: Very short tenure ({tenure} months) - new customers are more likely to churn")
        if MonthlyCharges > 80:
            insights.append(f"**Risk Factor**: High monthly charges (${MonthlyCharges:.1f}) - price sensitivity")
        if InternetService == "Fiber optic":
            insights.append("**Risk Factor**: Fiber optic customers tend to have higher churn rates")
        
        if insights:
            for insight in insights:
                st.markdown(f"• {insight}")
        else:
            st.markdown("• No significant risk factors identified")
        
        if pred == 1:
            st.warning("""
            **🚨 Immediate Retention Actions Recommended:**
            1. **Offer loyalty discount** (15-20% for 1-year commitment)
            2. **Personalized outreach** from customer success team
            3. **Service bundle** with added value features
            4. **Early renewal incentive** with gift card
            """)
        else:
            st.success("""
            **✅ Standard Monitoring Sufficient:**
            1. **Quarterly check-ins** for satisfaction
            2. **Cross-sell opportunities** for additional services
            3. **Referral program** invitation
            4. **Regular usage pattern monitoring**
            """)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        st.info("Please check that your model and data files are correctly configured.")


# Footer
st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align:center; color:var(--muted); font-size:13px; padding:20px; 
                border-top:1px solid rgba(255,255,255,0.05); margin-top:30px;'>
        Developed by Mrunal • Customer Churn ML Project
    </div>
    """,
    unsafe_allow_html=True
)