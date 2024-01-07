# Import our dependencies
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


st.write("Hello ,let's learn how to build a streamlit app together")

tabular_data = pd.read_csv('https://mlee22ph.github.io/Project4_Group11_AR_GP_ML/Resources/HAM10000_metadata.csv')
st.write(tabular_data.head())
