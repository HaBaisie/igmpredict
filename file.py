import streamlit as st
import pandas as pd
import numpy as np
import pickle

def load_models():
    with open("model_igg.pkl", "rb") as f:
        model_igg = pickle.load(f)
    with open("model_igm.pkl", "rb") as f:
        model_igm = pickle.load(f)
    return model_igg, model_igm

def user_input_features():
    st.sidebar.header('User Input Features')
    
    risk_factor = st.sidebar.selectbox('Risk Factor', (
        'Blood transfusion', 'Malaria Parasite', 'Typhoid', 'Residential area', 
        'Nearness to bush', 'Closeness to stagnant water or uncovered gutter', 
        'Use of Mosquito repellant?', 'Use of Mosquito Net'))
    value = st.sidebar.selectbox('Value', (
        'Yes', 'No', 'Rural', 'Urban', 'Frequently', 'Rarely', 'Never', 'Not close', 'Very close'))
    n = st.sidebar.number_input('n', min_value=0, max_value=200, step=1, value=50)
    or_igg = st.sidebar.number_input('OR IgG', value=1.0)
    ci_lower_igg = st.sidebar.number_input('CI Lower IgG', value=0.0)
    ci_upper_igg = st.sidebar.number_input('CI Upper IgG', value=1.0)
    p_value_igg = st.sidebar.number_input('p-value IgG', value=0.05)
    or_igm = st.sidebar.number_input('OR IgM', value=1.0)
    ci_lower_igm = st.sidebar.number_input('CI Lower IgM', value=0.0)
    ci_upper_igm = st.sidebar.number_input('CI Upper IgM', value=1.0)
    p_value_igm = st.sidebar.number_input('p-value IgM', value=0.05)
    
    data = {
        'n': n,
        'OR IgG': or_igg,
        'p-value IgG': p_value_igg,
        'OR IgM': or_igm,
        'p-value IgM': p_value_igm,
        'Value_Frequently': value == 'Frequently',
        'Value_Never': value == 'Never',
        'Value_No': value == 'No',
        'Value_Not close': value == 'Not close',
        'Value_Rarely': value == 'Rarely',
        'Value_Rural': value == 'Rural',
        'Value_Urban': value == 'Urban',
        'Value_Very close': value == 'Very close',
        'Value_Yes': value == 'Yes',
        'Risk Factor_Closeness to stagnant water or uncovered gutter': risk_factor == 'Closeness to stagnant water or uncovered gutter',
        'Risk Factor_Malaria Parasite': risk_factor == 'Malaria Parasite',
        'Risk Factor_Nearness to bush': risk_factor == 'Nearness to bush',
        'Risk Factor_Residential area': risk_factor == 'Residential area',
        'Risk Factor_Total': risk_factor == 'Total',
        'Risk Factor_Typhoid': risk_factor == 'Typhoid',
        'Risk Factor_Use of Mosquito Net': risk_factor == 'Use of Mosquito Net',
        'Risk Factor_Use of Mosquito repellant?': risk_factor == 'Use of Mosquito repellant?',
        'CI_Lower_IgG': ci_lower_igg,
        'CI_Upper_IgG': ci_upper_igg,
        'CI_Lower_IgM': ci_lower_igm,
        'CI_Upper_IgM': ci_upper_igm
    }
    features = pd.DataFrame(data, index=[0])
    return features

def preprocess_input(df):
    df = df[[
        'n', 'OR IgG', 'p-value IgG', 'OR IgM', 'p-value IgM', 'Value_Frequently',
        'Value_Never', 'Value_No', 'Value_Not close', 'Value_Rarely', 'Value_Rural', 
        'Value_Urban', 'Value_Very close', 'Value_Yes', 
        'Risk Factor_Closeness to stagnant water or uncovered gutter', 
        'Risk Factor_Malaria Parasite', 'Risk Factor_Nearness to bush', 
        'Risk Factor_Residential area', 'Risk Factor_Total', 'Risk Factor_Typhoid', 
        'Risk Factor_Use of Mosquito Net', 'Risk Factor_Use of Mosquito repellant?', 
        'CI_Lower_IgG', 'CI_Upper_IgG', 'CI_Lower_IgM', 'CI_Upper_IgM'
    ]]
    return df

def main():
    st.title("IgG and IgM Prediction")
    
    model_igg, model_igm = load_models()
    
    input_df = user_input_features()
    
    input_processed = preprocess_input(input_df)
    
    # Make predictions
    pred_igg = model_igg.predict(input_processed)
    pred_igm = model_igm.predict(input_processed)
    
    st.subheader('Prediction Results')
    st.write('Predicted IgG Positive (%):', pred_igg[0])
    st.write('Predicted IgM Positive (%):', pred_igm[0])

if __name__ == "__main__":
    main()
