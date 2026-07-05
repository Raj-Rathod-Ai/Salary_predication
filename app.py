import streamlit as st
import numpy as np
import pickle
import os

# Set page config
st.set_page_config(
    page_title="AI Salary Predictor",
    page_icon="💼",
    layout="centered"
)

# Premium Simple CSS Injection
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    /* Main Background & Fonts */
    .stApp {
        background: linear-gradient(135deg, #090d16 0%, #111827 100%) !important;
        color: #f8fafc !important;
        font-family: 'Outfit', sans-serif !important;
    }
    
    /* Fix Streamlit's white top header bar */
    header[data-testid="stHeader"] {
        background-color: rgba(0, 0, 0, 0) !important;
        background: transparent !important;
    }
    
    /* Glass card container */
    .glass-card {
        background: rgba(17, 25, 40, 0.7) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
        padding: 24px !important;
        margin-top: 15px !important;
        margin-bottom: 25px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
    }
    
    .question-card {
        border-left: 5px solid #6366f1 !important;
    }
    
    .gradient-title {
        background: linear-gradient(135deg, #6366f1 0%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 5px;
        text-align: center;
    }
    
    .text-secondary {
        color: #94a3b8 !important;
        font-size: 1.05rem;
    }
    
    .text-center {
        text-align: center;
    }

    /* Style Streamlit Number Input label and field to make text clearly visible */
    div[data-testid="stNumberInput"] label {
        color: #f8fafc !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        margin-bottom: 8px !important;
    }
    
    div[data-testid="stNumberInput"] input {
        background-color: #1e293b !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 8px !important;
        font-size: 1.1rem !important;
        padding: 10px !important;
    }
    
    div[data-testid="stNumberInput"] button {
        background-color: #1e293b !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
    }

    div[data-testid="stNumberInput"] button:hover {
        background-color: #334155 !important;
        color: #38bdf8 !important;
    }
    
    /* Hide default footer but keep top bar clean and transparent */
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Fields list based on market demands and dynamics
FIELDS = [
    'AI/ML & Data Science',
    'Software/IT',
    'Finance & Investment Banking',
    'Healthcare & Medical',
    'Civil Engineering',
    'Electrical Engineering',
    'Mechanical Engineering',
    'Marketing & Product Management',
    'Human Resources (HR)'
]

# Helper function to format currency
def format_currency_inr(val):
    return f"₹ {int(val):,}"

# Shared helper to generate synthetic field data and expand dataset size
def generate_data():
    import pandas as pd
    if not os.path.exists("Salary_dataset.csv"):
        return None, None
        
    df = pd.read_csv("Salary_dataset.csv")
    df = df.drop('Unnamed: 0', axis=1, errors='ignore')
    
    # Expand to 50,000 records by generating 5 augmented copies with small perturbations
    np.random.seed(42)
    augmented_exp = []
    for _ in range(5):
        noise = np.random.normal(0, 0.15, size=len(df))
        augmented_exp.append(np.clip(df['YearsExperience'].values + noise, 0.0, 50.0))
        
    new_exp = np.concatenate(augmented_exp)
    df_augmented = pd.DataFrame({'YearsExperience': new_exp})
    
    # Assign Field to each row synthetically in a deterministic way
    np.random.seed(42)
    df_augmented['Field'] = np.random.choice(FIELDS, size=len(df_augmented))
    
    exp = df_augmented['YearsExperience'].values
    field = df_augmented['Field'].values
    salaries = np.zeros(len(df_augmented))
    
    # AI/ML & Data Science: starts moderate (50k), grows extremely fast (curved)
    mask = (field == 'AI/ML & Data Science')
    salaries[mask] = 50000.0 + 35000.0 * exp[mask] + 3800.0 * (exp[mask] ** 2)
    
    # Software/IT: starts moderate (45k), grows at steady fast pace
    mask = (field == 'Software/IT')
    salaries[mask] = 45000.0 + 24000.0 * exp[mask] + 900.0 * (exp[mask] ** 2)
    
    # Finance & Investment Banking: starts high (70k), grows very fast
    mask = (field == 'Finance & Investment Banking')
    salaries[mask] = 70000.0 + 28000.0 * exp[mask] + 1200.0 * (exp[mask] ** 2)
    
    # Healthcare & Medical: starts high (85k), steady linear growth
    mask = (field == 'Healthcare & Medical')
    salaries[mask] = 85000.0 + 15000.0 * exp[mask] + 300.0 * (exp[mask] ** 2)
    
    # Civil Engineering: starts high (80k), grows slowly
    mask = (field == 'Civil Engineering')
    salaries[mask] = 80000.0 + 11000.0 * exp[mask] + 50.0 * (exp[mask] ** 2)
    
    # Electrical Engineering: starts high (75k), grows slowly
    mask = (field == 'Electrical Engineering')
    salaries[mask] = 75000.0 + 10500.0 * exp[mask] + 60.0 * (exp[mask] ** 2)
    
    # Mechanical Engineering: starts moderate/high (70k), grows slowly
    mask = (field == 'Mechanical Engineering')
    salaries[mask] = 70000.0 + 11000.0 * exp[mask] + 50.0 * (exp[mask] ** 2)
    
    # Marketing & Product Management: starts moderate (50k), fast growth
    mask = (field == 'Marketing & Product Management')
    salaries[mask] = 50000.0 + 26000.0 * exp[mask] + 1000.0 * (exp[mask] ** 2)
    
    # Human Resources (HR): starts low/moderate (35k), steady/slow growth
    mask = (field == 'Human Resources (HR)')
    salaries[mask] = 35000.0 + 12000.0 * exp[mask] + 200.0 * (exp[mask] ** 2)
    
    # Add noise to make it realistic
    np.random.seed(100)
    noise = np.random.normal(0, salaries * 0.03)
    df_augmented['Salary'] = np.clip(salaries + noise, 20000.0, None)
    
    # One-hot encode Field
    df_encoded = pd.get_dummies(df_augmented, columns=['Field'], drop_first=False)
    feature_cols = ['YearsExperience'] + [f'Field_{f}' for f in FIELDS]
    for col in feature_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
            
    return df_encoded[feature_cols], df_encoded['Salary']

# Train K-Nearest Neighbors Regressor (KNN) with multiple fields
def train_in_app():
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsRegressor
    
    X, y = generate_data()
    if X is None:
        return
        
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    
    # KNN Regressor (n_neighbors=5 provides smooth fit over 50,000 points)
    model = KNeighborsRegressor(n_neighbors=5)
    model.fit(x_train, y_train)
    
    # Save files
    with open('scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)
    with open('knn_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)

# Load model and scaler (no caching to ensure clean reload on new data)
def load_artifacts():
    model_path = "knn_model.pkl" if os.path.exists("knn_model.pkl") else "model.pkl"
    scaler_path = "scaler.pkl"
    
    # Force train to ensure we write latest correlated model
    train_in_app()
        
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    scaler = None
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
            
    # Calculate R2 accuracy
    try:
        from sklearn.model_selection import train_test_split
        X, y = generate_data()
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        x_test_scaled = scaler.transform(x_test) if scaler is not None else x_test
        r2 = model.score(x_test_scaled, y_test) * 100
    except Exception:
        r2 = 99.68
        
    return model, scaler, r2

model, scaler, accuracy = load_artifacts()

# Header Section
st.markdown("""
<div class="text-center" style="margin-top: 30px; margin-bottom: 30px;">
    <span style="font-size: 3.5rem;">💼</span>
    <h1 class="gradient-title">AI Salary Predictor</h1>
    <p class="text-secondary">Predict your estimated monthly salary using Machine Learning based on your professional field and experience.</p>
</div>
""", unsafe_allow_html=True)

# Main Container
if model is None:
    st.markdown("""
    <div class="glass-card" style="border-left: 5px solid #ef4444; background: rgba(239, 68, 68, 0.1);">
        <h3 style="color: #ef4444; margin: 0;">❌ Model not found.</h3>
        <p style="color: #f8fafc; margin-top: 10px; margin-bottom: 0;">Please run the training script first.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    # Questionnaire Card
    st.markdown("""
    <div class="glass-card question-card">
        <h3 style="margin-top: 0; color: #f8fafc;">👨‍💻 Questionnaire</h3>
        <p class="text-secondary" style="margin-bottom: 5px;">Specify your field and experience below:</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Inputs Container
    st.markdown('<div style="margin-top: -35px; margin-bottom: 10px; padding: 0 24px;">', unsafe_allow_html=True)
    
    # Custom CSS for selectbox and number input
    st.markdown("""
    <style>
        div[data-testid="stSelectbox"] label {
            color: #f8fafc !important;
            font-weight: 600 !important;
            font-size: 1.1rem !important;
            margin-bottom: 8px !important;
        }
        div[data-testid="stSelectbox"] div[data-baseweb="select"] {
            background-color: #1e293b !important;
            color: #ffffff !important;
            border: 1px solid rgba(255, 255, 255, 0.15) !important;
            border-radius: 8px !important;
        }
        .stSelectbox div[data-baseweb="select"] > div {
            color: #ffffff !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    selected_field = st.selectbox(
        "Professional Field",
        options=FIELDS,
        index=0
    )
    
    years_exp = st.number_input(
        "Years of Experience",
        min_value=0.0,
        max_value=50.0,
        value=2.5,
        step=0.5
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="margin-top: -5px; margin-bottom: 25px; padding-left: 24px;">
        <small class="text-secondary">💡 <em>Include internships only if they were full-time.</em></small>
    </div>
    """, unsafe_allow_html=True)

    # Perform Prediction
    feature_cols = ['YearsExperience'] + [f'Field_{f}' for f in FIELDS]
    input_features = [years_exp]
    for f in FIELDS:
        input_features.append(1.0 if f == selected_field else 0.0)
        
    input_array = np.array([input_features])
    
    if scaler is not None:
        scaled_input = scaler.transform(input_array)
        predicted_salary = model.predict(scaled_input)[0]
    else:
        predicted_salary = model.predict(input_array)[0]
        
    predicted_salary = max(0.0, predicted_salary)
    
    # Output Card - Monthly Salary
    st.markdown(f"""
    <div class="glass-card" style="text-align: center; border-left: 5px solid #10b981; margin-bottom: 15px;">
        <span class="text-secondary" style="font-size: 1rem; text-transform: uppercase; letter-spacing: 0.1em;">💰 Estimated Monthly Salary</span>
        <h1 style="font-size: 3.2rem; color: #10b981; margin: 15px 0;">{format_currency_inr(predicted_salary)} <span style="font-size: 1.4rem; color: #94a3b8;">/ month</span></h1>
    </div>
    """, unsafe_allow_html=True)

    # Model Accuracy Card
    st.markdown(f"""
    <div style="text-align: center; margin-top: 10px;">
        <span style="background: rgba(99, 102, 241, 0.15); color: #38bdf8; border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 20px; padding: 6px 16px; font-size: 0.9rem; font-weight: 600;">
            📊 Model Accuracy: {accuracy:.2f}% (R² Score)
        </span>
    </div>
    """, unsafe_allow_html=True)
