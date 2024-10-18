import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load trained model
model_path = 'xgboost_model.pkl'
model = joblib.load(model_path)

# Preprocess data function
def preprocess_data(df):
    # Define mappings for categorical variables
    mappings = {
        'poutcome': {'unknown': -1, 'failure': 0, 'success': 1},
        'contact': {'unknown': -1, 'telephone': 0, 'cellular': 1},
        'housing': {'no': 0, 'yes': 1},
        'loan': {'no': 0, 'yes': 1}
    }

    # Apply mappings
    df['poutcome'] = df['poutcome'].map(mappings['poutcome'])
    df['contact'] = df['contact'].map(mappings['contact'])
    df['housing'] = df['housing'].map(mappings['housing'])
    df['loan'] = df['loan'].map(mappings['loan'])
    
    # Convert 'duration' to numeric
    df['duration'] = pd.to_numeric(df['duration'], errors='coerce')
    df.fillna(0, inplace=True)  # Fill NaNs with 0 for simplicity
    
    return df

# Preprocess CSV data function (to handle object dtype columns)
def preprocess_data_csv(df):
    # Convert categorical columns to numeric using LabelEncoder
    categorical_columns = ['job', 'marital', 'education', 'default', 'month']
    label_encoders = {}
    
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    return preprocess_data(df)

def create_full_feature_set(df):
    # Define all features used by the model in the training phase
    features_order = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
                      'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays',
                      'previous', 'poutcome', 'lquarter', 'contacted']

    # Ensure all necessary features are present and in the correct order
    df = df.reindex(columns=features_order, fill_value=0)
    return df

# Streamlit UI for user input
st.title("Bank Customer Term Deposit Prediction")

# Form input prediction
with st.form("input_form"):
    duration = st.text_input("Duration (last contact duration)")
    poutcome = st.selectbox("Outcome of the previous marketing campaign", ["unknown", "failure", "success"])
    contact = st.selectbox("Contact communication type", ["unknown", "telephone", "cellular"])
    housing = st.selectbox("Has housing loan?", ["no", "yes"])
    loan = st.selectbox("Has personal loan?", ["no", "yes"])
    
    submitted = st.form_submit_button("Predict")
    if submitted:
        input_dict = {'duration': [duration], 'poutcome': [poutcome], 'contact': [contact],
                      'housing': [housing], 'loan': [loan]}
        input_df = pd.DataFrame(input_dict)
        processed_data = preprocess_data(input_df)
        features = create_full_feature_set(processed_data)
        prediction_prob = model.predict_proba(features)[:, 1][0]
        result = "Likely to subscribe" if prediction_prob > 0.5 else "Unlikely to subscribe"
        st.write(f"Prediction: {result} with probability {prediction_prob:.2f}")

# Handling batch predictions from CSV file uploads
st.subheader("Batch Prediction from CSV File")
uploaded_file = st.file_uploader("Upload your CSV file here.", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    processed_data = preprocess_data_csv(data)  # Process CSV data with categorical conversion
    features = create_full_feature_set(processed_data)
    predictions = model.predict_proba(features)[:, 1]
    
    # Add predictions to the data
    data['Probability'] = predictions
    high_prob = data[data['Probability'] > 0.5]
    low_prob = data[data['Probability'] <= 0.5]
    
    # Display results
    st.subheader("Customers Likely to Subscribe (>50% Probability)")
    st.write(high_prob)
    st.subheader("Customers Unlikely to Subscribe (≤50% Probability)")
    st.write(low_prob)
