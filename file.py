import streamlit as st
import pandas as pd
import pickle

# Load the saved models
with open("model_igg.pkl", "rb") as f:
    model_igg = pickle.load(f)

with open("model_igm.pkl", "rb") as f:
    model_igm = pickle.load(f)

def predict_igg(data):
    # Make prediction for IgG
    prediction_igg = model_igg.predict(data)
    return prediction_igg

def predict_igm(data):
    # Make prediction for IgM
    prediction_igm = model_igm.predict(data)
    return prediction_igm

def main():
    st.title("IgG and IgM Prediction App")

    # Sidebar
    st.sidebar.header("Input Parameters")

    # Get user input
    value = st.sidebar.radio("Blood Transfusion:", ["Yes", "No"])
    malaria_parasite = st.sidebar.radio("Malaria Parasite:", ["Yes", "No"])
    typhoid = st.sidebar.radio("Typhoid:", ["Yes", "No"])
    residential_area = st.sidebar.radio("Residential Area:", ["Rural", "Urban"])
    nearness_to_bush = st.sidebar.radio("Nearness to Bush:", ["Yes", "No"])
    closeness_to_water = st.sidebar.radio("Closeness to Water:", ["Not close", "Very close"])
    use_of_repellant = st.sidebar.radio("Use of Mosquito Repellant:", ["Frequently", "Rarely", "Never"])
    use_of_net = st.sidebar.radio("Use of Mosquito Net:", ["Yes", "No"])

    # Map user inputs to numerical values
    value_map = {"Yes": 1, "No": 0}
    malaria_parasite_map = {"Yes": 1, "No": 0}
    typhoid_map = {"Yes": 1, "No": 0}
    residential_area_map = {"Rural": 1, "Urban": 0}
    nearness_to_bush_map = {"Yes": 1, "No": 0}
    closeness_to_water_map = {"Not close": 1, "Very close": 0}
    use_of_repellant_map = {"Frequently": 2, "Rarely": 1, "Never": 0}
    use_of_net_map = {"Yes": 1, "No": 0}

    # Create a DataFrame for prediction
    input_data_igg = pd.DataFrame({
        "Risk Factor_Malaria Parasite": [malaria_parasite_map[malaria_parasite]],
        "Risk Factor_Typhoid": [typhoid_map[typhoid]],
        "Risk Factor_Residential area_Rural": [residential_area_map[residential_area]],
        "Risk Factor_Nearness to bush": [nearness_to_bush_map[nearness_to_bush]],
        "Risk Factor_Closeness to stagnant water or uncovered gutter_Not close": [closeness_to_water_map[closeness_to_water]],
        "Risk Factor_Use of Mosquito repellant?_Frequently": [use_of_repellant_map[use_of_repellant] == 2],
        "Risk Factor_Use of Mosquito repellant?_Rarely": [use_of_repellant_map[use_of_repellant] == 1],
        "Risk Factor_Use of Mosquito repellant?_Never": [use_of_repellant_map[use_of_repellant] == 0],
        "Risk Factor_Use of Mosquito Net_Yes": [use_of_net_map[use_of_net]],
        "Value_Yes": [value_map[value]],
        "95% CI IgG": [0],
        "95% CI IgM": [0],
        "OR IgG": [0],
        "OR IgM": [0],
        "p-value IgG": [0],
        "p-value IgM": [0],
        "Risk Factor_Total": [0],
        "Value_Frequently": [0],
        "Value_Never": [0],
        "Value_No": [0],
        "Value_Not close": [0]
    })
    input_data_igm = input_data_igg.copy()  # Create a copy of input_data_igg for IgM prediction

    # Get feature names
    feature_names_igg = input_data_igg.columns
    feature_names_igm = input_data_igm.columns

    # Make predictions
    prediction_igg = predict_igg(input_data_igg)
    prediction_igm = predict_igm(input_data_igm)

    st.write("### Prediction Results:")
    st.write("#### IgG Positive Percentage:", prediction_igg[0])
    st.write("#### IgM Positive Percentage:", prediction_igm[0])

if __name__ == '__main__':
    main()
