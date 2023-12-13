import streamlit as st
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

st.markdown("# Braden's App: LinkedIn Usage Predictor")

st.markdown("## This app will predict if an individual is a LinkedIn user based on the following inputs:")

income_labels = ["Less than $10k", "$10k to $20k" , "$20k to $30k" , "$30k to $40k", "$40k to $50k", "$50k to $75k" , "$75k to $100k" , "$100k to $150k", "$150k or more"]
inc = st.selectbox("Income Level:", income_labels)
def income_level(inc): 
    if inc == "Less than $10k": 
        income = 1
    elif inc == "$10k to $20k": 
        income = 2
    elif inc == "$20k to $30k": 
        income = 3
    elif inc == "$30k to $40k": 
        income = 4
    elif inc == "$40k to $50k": 
        income = 5
    elif inc == "$50k to $75k": 
        income = 6
    elif inc == "$75k to $100k": 
        income = 7
    elif inc == "$100k to $150k": 
        income = 8
    elif inc == "$150k or more": 
        income = 9
    else: 
        income = 0
    return income

income = income_level(inc)

st.write("Income level is", income)

education_labels = ["Less than high school", "High school incomplete" , "High school graduate (or GED)" , "Some college (no degree)", "Two-year associate degree", "Four-year bachelor's degree" , "Some postgraduate or professional schooling (no degree)" , "Postgraduate degree"]
edu = st.selectbox("Highest level of school/degree completed:", education_labels)
def education_level(edu): 
    if edu == "Less than high school": 
        education = 1
    elif edu == "High school incomplete": 
        education = 2
    elif edu == "High school graduate (or GED)": 
        education = 3
    elif edu == "Some college (no degree)": 
        education = 4
    elif edu == "Two-year associate degree": 
        education = 5
    elif edu == "Four-year bachelor's degree": 
        education = 6
    elif edu == "Some postgraduate or professional schooling (no degree)": 
        education = 7
    elif edu == "Postgraduate degree": 
        education = 8
    else: 
        education = 0
    return education

education = education_level(edu)

st.write("Education level is", education)

parent_labels = ["Yes", "No"]
par = st.selectbox("Are you a parent of a child under 18 living in your home?", parent_labels)

def parent_func(par): 
    if par == "Yes":
        Parent = 1
    elif par == "No":
        Parent = 0
    else: 
        Parent = 0
    return (Parent)

parent = parent_func(par)

st.write("Parent value is", parent)

married_labels = ["Yes", "No"]
mar = st.selectbox("Are you married", married_labels)

def married_func(par): 
    if mar == "Yes":
        married = 1
    elif mar == "No":
        married = 0
    else: 
        married = 0
    return (married)

married = married_func(mar)

st.write("Marriage value is", married)

gender_labels = ["Male", "Female", "Other/Prefer not to answer"]
gen = st.selectbox("What is your gender?", gender_labels)

def female_func(gen): 
    if gen == "Female":
        female = 1
    elif gen == "Male":
        female = 0
    else: 
        female = 0
    return (female)

female = female_func(gen)

st.write("Gender value is", female)

age = st.text_input("What is your current age?", value = "1-98")

st.write("Age is", age)

s = pd.read_csv("social_media_usage.csv")

def clean_sm(x):
    x =  np.where(x == 1, 1, 0)
    return x

def clean_gender(x):
    x = np.where(x == 2, 1, 0)
    return x

ss = pd.DataFrame(
    {"sm_li": clean_sm(s.web1h),
     "income": np.where(s.income <= 9, s.income, np.nan),
     "education": np.where(s.educ2 <= 8, s.educ2, np.nan),
     "parent": clean_sm(s.par),
     "married": clean_sm(s.marital),
     "female": clean_gender(s.gender),
     "age": np.where(s.age <= 98, s.age, np.nan)
    })

ss = ss.dropna()

x = ss[["income", "education", "parent", "married", "female", "age"]]
y = ss ["sm_li"]

train_x, test_x, train_y, test_y = train_test_split(x, y, shuffle=True, test_size=0.2)

log_reg = LogisticRegression(class_weight = "balanced")
log_reg.fit(train_x, train_y)

user_df = pd.DataFrame({"income": [income],
                        "education": [education],
                        "parent": [parent],
                        "married": [married], 
                        "female": [female], 
                        "age": [age]
                        })

def prediction(user_df):
    prediction_user = log_reg.predict(user_df)
    return  prediction_user

prediction_user = prediction(user_df)

def prediction_func(prediction_user):
    if prediction_user == 1:
        final = "a LinkedIn User"
    elif prediction_user == 0: 
        final = "not a LinkedIn User"
    return(final)
    
final = prediction_func(prediction_user)

st.write("The model predicts that you are", final)

def prediction_prob(user_df):
    prediction_p = log_reg.predict_proba(user_df)
    return  prediction_p * 100

prediction_p = prediction_prob(user_df)

st.write("The probability that you are a LinkedIn user is (%):", prediction_p[0,1])





