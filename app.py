import streamlit as st
import pickle
import numpy as np
import pandas as pd  # Add this import

# Load the model and dataframe
pipe = pickle.load(open("pipe.pkl", "rb"))
df = pickle.load(open("df.pkl", "rb"))

st.title("Laptop Price Predictor")

# All your existing input widgets remain the same
company = st.selectbox("Brand", df["Company"].unique())
type = st.selectbox("Type", df["TypeName"].unique())
ram = st.selectbox("RAM (in GB)", [2, 4, 6, 8, 12, 16, 24, 32, 64])
weight = st.number_input("Weight (in kg)", min_value=0.1, max_value=10.0, step=0.1)
touchscreen = st.selectbox("Touchscreen", ["No", "Yes"])
ips = st.selectbox("IPS", ["No", "Yes"])
screen_size = st.number_input("Screen Size (in inches)", min_value=10.0, max_value=20.0, step=0.1)
resolution = st.selectbox('Screen Resolution', [
    '1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', 
    '2560x1600', '2560x1440', '2304x1440'
])
cpu = st.selectbox('CPU', df['Cpu brand'].unique())
hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])
gpu = st.selectbox('GPU', df['GPU_Brand'].unique())
os = st.selectbox('OS', df['os'].unique())

if st.button('Predict Price'):
    # Convert Touchscreen and IPS to binary
    touchscreen_binary = 1 if touchscreen == 'Yes' else 0
    ips_binary = 1 if ips == 'Yes' else 0

    # Extract resolution dimensions and compute PPI
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size

    # Create a dictionary with the feature names and values
    query_dict = {
        'Company': company,
        'TypeName': type,
        'Ram': ram,
        'Weight': weight,
        'Touchscreen': touchscreen_binary,
        'Ips': ips_binary,
        'ppi': ppi,
        'Cpu brand': cpu,
        'HDD': hdd,
        'SSD': ssd,
        'GPU_Brand': gpu,
        'os': os
    }

    # Convert the dictionary to a pandas DataFrame
    query_df = pd.DataFrame([query_dict])

    try:
        # Make prediction
        predicted_price = np.exp(pipe.predict(query_df)[0])
        st.title(f"The predicted price of this configuration is ₹{int(predicted_price)}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
        
        # Debugging information
        st.write("Debug Information:")
        st.write("Query DataFrame columns:", query_df.columns.tolist())
        st.write("Expected features:", pipe.named_steps['preprocessor'].get_feature_names_out().tolist())

