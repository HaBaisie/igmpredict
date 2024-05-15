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
    
    risk_factor = st.sidebar.selectbox('Risk Factor', ('Blood transfusion', 'Malaria Parasite', 'Typhoid', 'Residential area', 'Nearness to bush', 'Closeness to stagnant water or uncovered gutter', 'Use of Mosquito repellant?', 'Use of Mosquito Net'))
    value = st.sidebar.selectbox('Value', ('Yes', 'No', 'Rural', 'Urban', 'Frequently', 'Rarely', 'Never', 'Not close', 'Very close'))
    n = st.sidebar.number_input('n', min_value=0, max_value=200, step=1, value=50)
    or_igg = st.sidebar.number_input('OR IgG', value=1.0)
    ci_lower_igg = st.sidebar.number_input('CI Lower IgG', value=0.0)
    ci_upper_igg = st.sidebar.number_input('CI Upper IgG', value=1.0)
    p_value_igg = st.sidebar.number_input('p-value IgG', value=0.05)
    or_igm = st.sidebar.number_input('OR IgM', value=1.0)
    ci_lower_igm = st.sidebar.number_input('CI Lower IgM', value=0.0)
    ci_upper_igm = st.sidebar.number_input('CI Upper IgM', value=1.0)
    p_value_igm = st.sidebar.number_input('p-value IgM', value=0.05)
    
    data = {'Risk Factor': risk_factor,
            'Value': value,
            'n': n,
            'OR IgG': or_igg,
            'CI_Lower_IgG': ci_lower_igg,
            'CI_Upper_IgG': ci_upper_igg,
            'p-value IgG': p_value_igg,
            'OR IgM': or_igm,
            'CI_Lower_IgM': ci_lower_igm,
            'CI_Upper_IgM': ci_upper_igm,
            'p-value IgM': p_value_igm}
    features = pd.DataFrame(data, index=[0])
    return features

def preprocess_input(df):
    # Convert categorical variables into dummy/indicator variables
    df = pd.get_dummies(df, columns=['Value', 'Risk Factor'], drop_first=True)
    return df

def main():
    st.title("IgG and IgM Prediction")
    
    # Load models
    model_igg, model_igm = load_models()
    
    # Get user input
    input_df = user_input_features()
    
    # Preprocess user input
    input_processed = preprocess_input(input_df)
    
    # Make sure to use the same feature order and names as during training
    model_features = pd.DataFrame(columns=['n', 'OR IgG', 'CI_Lower_IgG', 'CI_Upper_IgG', 'p-value IgG', 'OR IgM', 'CI_Lower_IgM', 'CI_Upper_IgM', 'p-value IgM',
                                           'Value_No', 'Value_Rural', 'Value_Urban', 'Value_Frequently', 'Value_Rarely', 'Value_Never',
                                           'Value_Not close', 'Value_Very close',
                                           'Risk Factor_Blood transfusion', 'Risk Factor_Malaria Parasite', 'Risk Factor_Typhoid', 'Risk Factor_Residential area', 'Risk Factor_Nearness to bush', 
                                           'Risk Factor_Closeness to stagnant water or uncovered gutter', 'Risk Factor_Use of Mosquito repellant?', 'Risk Factor_Use of Mosquito Net'])
    
    input_processed = pd.concat([input_processed, model_features]).fillna(0).loc[:, model_features.columns]

    st.write("Processed input aligned with model features:")
    st.write(input_processed)
    
    # Make predictions
    pred_igg = model_igg.predict(input_processed)
    pred_igm = model_igm.predict(input_processed)
    
    st.subheader('Prediction Results')
    st.write('Predicted IgG Positive (%):', pred_igg[0])
    st.write('Predicted IgM Positive (%):', pred_igm[0])

if __name__ == "__main__":
    main()
