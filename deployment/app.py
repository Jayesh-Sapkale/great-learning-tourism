import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Visit with Us - Wellness Tourism Predictor",
    page_icon="🏖️",
    layout="wide"
)

st.title("'Visit with Us' — Wellness Tourism Package Predictor")
st.markdown(
    "Automated MLOps Decision Support System to predict customer propensity "
    "to purchase the newly introduced **Wellness Tourism Package** before launching sales outreach."
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.joblib")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}. Ensure the CI/CD pipeline has trained and pushed the model.")
        return None
    return joblib.load(MODEL_PATH)

pipeline = load_model()

st.sidebar.header("Customer Profile Inputs")

with st.sidebar:
    age = st.slider("Age", 18, 70, 35)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    city_tier = st.selectbox("City Tier", [1, 2, 3], index=0)
    monthly_income = st.number_input("Monthly Income (₹)", min_value=1000.0, max_value=150000.0, value=25000.0, step=1000.0)

st.subheader("✈️ Travel & Interaction Details")
col1, col2, col3 = st.columns(3)

with col1:
    typeof_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
    duration_of_pitch = st.slider("Duration of Pitch (minutes)", 1, 120, 15)

with col2:
    num_trips = st.slider("Annual Number of Trips", 0, 25, 3)
    num_person = st.slider("Number of Persons Visiting", 1, 10, 3)
    num_children = st.slider("Children Below Age 5 Visiting", 0, 5, 1)

with col3:
    pref_star = st.selectbox("Preferred Hotel Rating (Stars)", [3.0, 4.0, 5.0], index=0)
    pitch_satisfaction = st.select_slider("Pitch Satisfaction Score", options=[1, 2, 3, 4, 5], value=3)
    num_followups = st.slider("Number of Follow-ups", 1, 10, 4)
    passport = st.radio("Valid Passport Holder?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    own_car = st.radio("Owns a Car?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# Build inference DataFrame matching exact schema
input_dict = {
    'Age': [age],
    'TypeofContact': [typeof_contact],
    'CityTier': [city_tier],
    'DurationOfPitch': [duration_of_pitch],
    'Occupation': [occupation],
    'Gender': [gender],
    'NumberOfPersonVisiting': [num_person],
    'NumberOfFollowups': [num_followups],
    'ProductPitched': [product_pitched],
    'PreferredPropertyStar': [pref_star],
    'MaritalStatus': [marital_status],
    'NumberOfTrips': [num_trips],
    'Passport': [passport],
    'PitchSatisfactionScore': [pitch_satisfaction],
    'OwnCar': [own_car],
    'NumberOfChildrenVisiting': [num_children],
    'Designation': [designation],
    'MonthlyIncome': [monthly_income]
}

input_df = pd.DataFrame(input_dict)

st.divider()
st.subheader("Input Feature Vector Preview")
st.dataframe(input_df, use_container_width=True)

if st.button("Predict Purchase Propensity", type="primary"):
    if pipeline is not None:
        prediction = pipeline.predict(input_df)[0]
        prob = pipeline.predict_proba(input_df)[0][1]
        
        st.subheader("Inference Result")
        if prediction == 1:
            st.success(f"**High Propensity to Buy!** \n\nProbability of Purchase: **{prob * 100:.2f}%**\n\n**Action:** Prioritize this customer for sales outreach.")
        else:
            st.warning(f"**Low Propensity to Buy.** \n\nProbability of Purchase: **{prob * 100:.2f}%**\n\n**Action:** Target with nurturing awareness campaigns before sales calls.")