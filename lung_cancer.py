import streamlit as st
import numpy as np
import pickle

# Load the trained model from a file
@st.cache_resource
def load_model():
    try:
        with open("lung_cancer.sav", "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("The model file was not found. Make sure 'lung_cancer.sav' is in the same folder.")
        return None

model = load_model()

# Mappings from text to numbers
gender_map = {'Male': 0, 'Female': 1}
country_map = {
    'Malta': 0, 'Ireland': 1, 'Portugal': 2, 'France': 3, 'Sweden': 4, 'Croatia': 5, 'Greece': 6, 'Spain': 7,
    'Netherlands': 8, 'Denmark': 9, 'Slovenia': 10, 'Belgium': 11, 'Hungary': 12, 'Romania': 13, 'Poland': 14,
    'Italy': 15, 'Germany': 16, 'Estonia': 17, 'Czech Republic': 18, 'Lithuania': 19, 'Slovakia': 20,
    'Austria': 21, 'Finland': 22, 'Luxembourg': 23, 'Cyprus': 24, 'Latvia': 25, 'Bulgaria': 26
}
stage_map = {'Stage III': 0, 'Stage IV': 1, 'Stage I': 2, 'Stage II': 3}
family_history_map = {'No': 0, 'Yes': 1}
smoke_map = {'Passive Smoker': 0, 'Never Smoked': 1, 'Former Smoker': 2, 'Current Smoker': 3}
treatment_map = {'Chemotherapy': 0, 'Surgery': 1, 'Combined': 2, 'Radiation': 3}

# Predict using the model
def predict_risk(features):
    data = np.array(features).reshape(1, -1)
    result = model.predict(data)[0]

    try:
        confidence = model.predict_proba(data).max()
        confidence_text = f" (Confidence: {confidence*100:.2f}%)"
    except:
        confidence_text = ""

    if result == 0:
        return f"High Risk of Lung Cancer{confidence_text}"
    else:
        return f"Low Risk of Lung Cancer{confidence_text}"

# Streamlit app interface
def main():
    st.title("Lung Cancer Risk Predictor")
    st.write("Fill in the details below to get a prediction.")

    st.subheader("Basic Information")
    age = st.number_input("Age", 1, 120, step=1)
    gender = st.selectbox("Gender", list(gender_map.keys()))
    country = st.selectbox("Country", list(country_map.keys()))

    st.subheader("Medical Information")
    stage = st.selectbox("Cancer Stage", list(stage_map.keys()))
    fam_history = st.selectbox("Family History of Cancer?", list(family_history_map.keys()))
    smoker = st.selectbox("Smoking Status", list(smoke_map.keys()))
    bmi = st.number_input("BMI", 10.0, 50.0, step=0.1)
    cholesterol = st.number_input("Cholesterol Level", 100, 300)

    # Binary health conditions
    hypertension = st.radio("Hypertension", [0, 1], format_func=lambda x: "Yes" if x else "No")
    asthma = st.radio("Asthma", [0, 1], format_func=lambda x: "Yes" if x else "No")
    cirrhosis = st.radio("Cirrhosis", [0, 1], format_func=lambda x: "Yes" if x else "No")
    other_cancer = st.radio("Other Cancers Diagnosed", [0, 1], format_func=lambda x: "Yes" if x else "No")

    treatment = st.selectbox("Treatment Type", list(treatment_map.keys()))

    if st.button("Check Risk"):
        if not model:
            return

        # Prevent prediction using untouched defaults
        if age == 1 and bmi == 10.0 and cholesterol == 100:
            st.warning("Please enter real values instead of defaults.")
            return

        try:
            inputs = [
                age,
                gender_map[gender],
                country_map[country],
                stage_map[stage],
                family_history_map[fam_history],
                smoke_map[smoker],
                bmi,
                cholesterol,
                hypertension,
                asthma,
                cirrhosis,
                other_cancer,
                treatment_map[treatment]
            ]

            if len(inputs) != 13:
                st.error("Error: Input data mismatch.")
                return

            prediction = predict_risk(inputs)
            st.success(prediction)
            st.info("This is just a prediction. For medical advice, consult a doctor.")

        except Exception as e:
            st.error(f"Something went wrong: {e}")

# Run the app
if __name__ == "__main__":
    main()
