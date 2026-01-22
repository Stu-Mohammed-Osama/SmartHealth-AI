import streamlit as st
import pandas as pd
import numpy as np
import joblib
model = joblib.load('insurance_model.pkl')
encoder = joblib.load('gender_encoder.pkl')
#background #F9F9F9
#the main button #3E1A5B
#options buttons #F0E5FF
#form color #FFFFFF
#main text color #342258
#mark tone text color #B5B4CA
#Index(['age', 'bmi', 'children', 'smoker', '0', '1'], dtype='object')
# Custom CSS for Streamlit buttons
st.set_page_config(layout="wide")
st.markdown("""
    <style>
    /* 1. Title (st.title) */
    [data-testid="stHeaderElement"] h1, 
    .stApp h1 {
        color: #9365ed !important;
        font-family: 'Arial' !important;
    }

    /* 2. Header (st.header) */
    [data-testid="stHeaderElement"] h2,
    .stApp h2 {
        color: #9365ed !important;
    }

    /* 3. Subheader (st.subheader) */
    [data-testid="stHeaderElement"] h3,
    .stApp h3 {
        color: #FFFFFF!important;
    }

    /* 4. Form Submit Button (st.form_submit_button) */
    /* This targets buttons inside the form specifically */
    div.stForm [data-testid="stBaseButton-secondaryFormSubmit"] {
        background-color: #F0E5FF;
        color: #4B0082;
        border: 2px solid #9370DB;
        border-radius: 10px;
        width: 100%;
    }
    
    /* Hover effect for the submit button */
    div.stForm [data-testid="stBaseButton-secondaryFormSubmit"]:hover {
        background-color: #3E1A5B;
        color: white;
    }
    .stApp {
    background-color: #0e1117;
    }
    section[data-testid="stSidebar"] {
    background-color: #161a1f;
    }
     /* ===== Markdown text ===== */
    .stMarkdown, .stMarkdown p {
        color: #dcdcdc;
        font-size: 16px;
    }
    .st-emotion-cache-14vh5up{
         background-color: #0e1117;  
        }
            
    .st-emotion-cache-1r4qj8v{
            color : white;
        }
    
    .st-emotion-cache-5r6ut5{
            color : rgba(250, 250, 250, 0.6);
        }
    
    </style>
""", unsafe_allow_html=True)
col1 , col2 = st.columns([0.6, 0.4] , gap="xlarge" , vertical_alignment="center")
charges = 0.00
with col1:
    with st.form("the medical form" , height="content"):
        st.subheader("Write the information , please")
        sex = st.selectbox("Your gender" , ("male" , "female"))
        bmi = st.number_input("put th BMI" , min_value=16 , value=25 , max_value=200)
        age = st.number_input("put your age",min_value= 1 , max_value=120 , value=14)
        children = st.number_input("the number of children" , min_value=0)
        smoker = st.selectbox("Are you smoker" ,("no" , "yes"))
        predict = st.form_submit_button("PREDICT!", width= "stretch")
        if predict:
          if age < 18 and children > 0:
            st.warning("write logical information")
          else:
            smoker_val= 1 if smoker == "yes" else 0
            sex_encoded = encoder.transform([[sex]])
            input_data = [[age, bmi, children , smoker_val, sex_encoded[0][0], sex_encoded[0][1]]]
            charges = round(model.predict(input_data)[0] , 3)

with col2:
     st.header("the prediction of the health insurance cost in US :" , text_alignment="center")
     st.title(str(charges) + "$" , text_alignment="center")
     st.image("ChatGPT Image Jan 21, 2026, 06_04_06 PM.png")

with st.sidebar:
   st.subheader("Developer Information :")
   st.markdown("**name : Mohammed Osama**")
   st.markdown("**school : Mohammed Othman**")
   st.markdown("**grade : 12**")
   st.markdown("**country : Egypt**")

   st.markdown("**dream major : Computer Science**")
