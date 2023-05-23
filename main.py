import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd 
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import NearMiss


dataset = pd.read_csv("waterQuality1.csv") 

missing_value = ['#NUM!', np.nan]
df=pd.read_csv("waterQuality1.csv", na_values = missing_value)

df['ammonia'] = pd.to_numeric(df['ammonia'])
df['is_safe'] = pd.to_numeric(df['is_safe'])

df.dropna( subset=['ammonia', 'is_safe'], axis=0, inplace=True)

x = df.iloc[:,:-1]
y = df.iloc[:,-1]

nm = NearMiss()
x_resample, y_resample = nm.fit_resample(x, y)

xtrain, xtest, ytrain, ytest = train_test_split(x_resample, y_resample, test_size = 0.2, random_state = 0)


rfcl = RandomForestClassifier(n_estimators=200, criterion='gini', min_samples_split=5, min_samples_leaf=2, max_features='auto', bootstrap=True, n_jobs=-1, random_state=42)
rfcl.fit(xtrain, ytrain)


# web title
st.set_page_config(
    page_title="Water Q",
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
    st.write("# Water Q")
    st.write(
    """
    Built on supervised machine learning algorithm  \n for classification called **Random Forest Classifier**.
    """
    )

    image1 = Image.open('random-forest.png')
    image2 = Image.open('one-water.png')
    image2.thumbnail((400,400))
    st.image(image2)
    st.image(image1)
    
    st.markdown(
    """
    - [Source Code](https://github.com/zeinrivo/waterq_app)
    """
    )

    st.caption("Created by **Zein Rivo**")


# option : Demo 
if selected == "Demo":
    st.title("Water Q")
    st.write("Customize the input below")

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
      result = rfcl.predict(x_new)
      if result == 0:
        st.subheader("Non Potable")
      if result == 1:
        st.subheader("Potable")
