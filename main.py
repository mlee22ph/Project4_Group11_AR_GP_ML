# Import our dependencies
import streamlit as st
import pandas as pd



st.write("Hello ,let's learn how to build a streamlit app together")

tabular_data = pd.read_csv('https://mlee22ph.github.io/Project4_Group11_AR_GP_ML/Resources/HAM10000_metadata.csv')
st.write(tabular_data.head())
