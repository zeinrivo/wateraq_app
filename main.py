import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd 
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from imblearn.under_sampling import NearMiss


# import dataset
dataset = pd.read_csv("waterQuality1.csv") 


# data prepro
missing_value = ['#NUM!', np.nan]
df=pd.read_csv("waterQuality1.csv", na_values = missing_value)

df['ammonia'] = pd.to_numeric(df['ammonia'])
df['is_safe'] = pd.to_numeric(df['is_safe'])

df.dropna( subset=['ammonia', 'is_safe'], axis=0, inplace=True)


# dataset splitting
x = df.iloc[:,:-1]
y = df.iloc[:,-1]

nm = NearMiss()
x_resample, y_resample = nm.fit_resample(x, y)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state = 0)


# train the model
lgbm = LGBMClassifier()
lgbm.fit(xtrain, ytrain)


# web title
st.set_page_config(
    page_title="Water Q Prediction ",
)


# navigation/option
with st.sidebar:
   selected = option_menu(
        menu_title="Main Menu",  
        options=["Home", "Demo"], 
        icons=["house", "record-circle"],  
        menu_icon="cast",  # optional
        default_index=0,  # optional         
)


# option : Home
if selected == "Home":
    st.write("# Water Q Prediction")
    st.write(
    """
    Built with supervised machine learning algorithm  \n for classification problem called **LightGBM**.
    """
    )

    image1 = Image.open('Captureqefq.JPG')
    
    st.image(image1)
    
    st.markdown(
    """
    - [Source Code](https://github.com/zeinrivo/lec-app)
    """
    )

    st.caption("Created by **Zein Rivo**")


# option : Demo 
if selected == "Demo":
    st.title("Water Q Prediction")
    st.write("Customize the input below with your personal data")

    aluminium = st.number_input("Aluminium")
    ammonia = st.number_input("Ammonia")
    arsenic = st.number_input("Arsenic")
    barium = st.number_input("Barium")
    cadmium = st.number_input("Cadmium")
    chloramine = st.number_input("Chloramine")
    chromium = st.number_input("Chromium")
    copper = st.number_input("Copper")
    flouride = st.number_input("Flouride")
    bacteria = st.number_input("Bacteria")
    viruses = st.number_input("Viruses")
    lead = st.number_input("Lead")
    nitrates = st.number_input("Nitrates")
    nitrites = st.number_input("Nitrites")
    mercury = st.number_input("Mercury")
    perchlorate = st.number_input("Perchlorate")
    radium = st.number_input("Radium")
    selenium = st.number_input("Selenium")
    silver = st.number_input("Silver")
    uranium = st.number_input("Uranium")
    
    ok = st.button ("Check Quality")

    if ok:
      x_new = [[aluminium, ammonia, arsenic, barium, cadmium, chloramine,chromium, copper, flouride, bacteria, viruses, lead,nitrates, nitrites, mercury, perchlorate, radium, selenium,silver, uranium]]
      result = lgbm.predict(x_new)
      if result == 0:
        st.subheader("Non Potable")
      if result == 1:
        st.subheader("Potable")
