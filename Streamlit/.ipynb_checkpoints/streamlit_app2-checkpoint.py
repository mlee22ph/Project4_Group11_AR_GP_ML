# Dependencies & Installs
import pandas as pd
import streamlit as st
from joblib import load
from PIL import Image
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder, StandardScaler

local_file_path = "Resources/cleaned_skin_metadata.csv"
df = pd.read_csv(local_file_path)
    
# Define Streamlit app
def app():

    # Load logo image
    logo_img = Image.open("Images/skin_cancer_1.jpg")
 
    # Expandable Sidebar
    st.sidebar.title("Navigation")
    st.sidebar.write("Expand the sections below to access different parts of the app.")

    # Create an expandable sidebar for additional information
    with st.sidebar.expander("Further reading if you are interested"):
        st.write("Read More about Skin Cancers here")
        st.write("Different types of Skin Cancers discussed here:")
        st.write("• [Actinic keratoses](https://www.skincancer.org/skin-cancer-information/actinic-keratosis/)")
        st.write("• [Basal cell carcinoma](https://www.acrf.com.au/support-cancer-research/types-of-cancer/basal-cell-skin-cancer/?psafe_param=1&utm_source=google_grant&utm_medium=cpc&utm_campaign={campaign}&utm_content=154819747361&utm_term=&gad_source=1&gclid=EAIaIQobChMI67KI1PnMgwMVVNBMAh2lfABnEAAYAiAAEgIo4PD_BwE)")
        st.write("• [Benign keratosis-like lesions](https://www.mayoclinic.org/diseases-conditions/seborrheic-keratosis/symptoms-causes/syc-20353878)")
        st.write("• [Dermatofibroma](https://www.theskincancerdoctor.com.au/education/skincancerlesions/dermatofibroma/)")
        st.write("• [Melanocytic nevi](https://www.dermcoll.edu.au/atoz/congenital-melanocytic-naevi/)")
        st.write("• [Melanoma](https://melanoma.org.au/about-melanoma/what-is-melanoma/?gclid=EAIaIQobChMIvryno_XMgwMVqswWBR1nJAC2EAAYASAAEgLiG_D_BwE)")
        st.write("• [Vascular lesions](https://beautyonrose.com.au/skin-condition-vascular-lesions-redness/)")

    # with st.sidebar.expander("Further reading if you are interested"):
    #     st.write("Click here for more reading materials")

    with st.sidebar.expander("Dataset used for modelling"):        
        st.write("[Dataset Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000/data)")
        st.write("[Dataset Harvard](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)")

    with st.sidebar.expander("Group Members"):
        st.write("• Athira")
        st.write("• Geoff")
        st.write("• Mike")

    # GitHub Repository section
    github_expander = st.sidebar.expander("GitHub Repository Section")
    with github_expander:
        st.markdown(
            "[Click here](https://github.com/mlee22ph/Project4_Group11_AR_GP_ML/tree/AR_branch) to visit the GitHub repository for this project."
        )
    
    
    # Create an empty container for the header
    header = st.empty()
    # Add logo image and app name to the header using Markdown
    header.markdown(
        f"""
        <div style="display: flex; align-items: center;">
            <img src="data:image/png;base64,{image_to_base64(logo_img)}" style="height: 50px; margin-right: 10px;" />
            <h1 style="margin: 0;">Skin Cancer Prediction App</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    
    ###############################################################################################################################################################

     # Load the trained models and scalers
    svm_model = load('Model1/svm_model.joblib')
    svm_scaler = load('Model1/svm_scaler.joblib')
    svm_label_encoder = load('Model1/svm_label_encoder.joblib')
    dx_type_label_encoder = load('Model1/dx_type_label_encoder.joblib')
    diagnosis_label_encoder = load('Model1/diagnosis_label_encoder.joblib')
    age_scaler = load('Model1/age_scaler.joblib')

    with st.container():
        st.write("This model will integrate four critical parameters: diagnostic type, patient age, patient sex, and lesion location.")
        st.write("\nSuch a model aims to assist in early-stage diagnosis, potentially improving patient outcomes.")
        st.write("\nDeveloped for academic purpose, not a substitute for professional medical advice and diagnosis")
    
    additional_text = """
    This application has been created to fulfill the requirements of the Data Analytics Boot Camp hosted by UWA in 2023 and should not be interpreted as medical advice. 
    Working with a limited dataset, time and expertise the information provided by this app may not be entirely accurate, as the primary intention was to implement and demonstrate the skills learned during the course. The focus was more on skill application rather than ensuring the accuracy of the data predictions. 
    This project is primarily meant for exploring and showcasing the student's knowledge, rather than providing reliable medical analysis or advice. 
    """
        
    # Create two columns for the layout
    left_column, right_column = st.columns(2)

    ###############################################################################################################################################################

    # User Inputs
    
    dx_type = st.selectbox('Select Diagnosis Type', options=['Histo', 'Follow_up','Confocal','Consensus'], help='The type of method used to identify the leision. The abbrevations are (histo) histopathology , (follow_up) follow-up examination, (consensus) expert consensus , (confocal) confocal microscopy.') 
    age = st.number_input('Enter Age', min_value=0, max_value=100, value=30, format='%i', key='age', help='Age in years')
    
    sex = st.selectbox('Select Sex', options=['Male', 'Female'], help='Male or Female')
    localization = st.selectbox('Select Localization', options=['Abdomen', 'Back', 'Chest', 'Ear','Face', 'Foot', 
                                                                'Hand', 'Lower extremity','Neck', 'Scalp', 'Trunk','Upper extremity']
                                                                , help='The location of the leision on your body') 


    # Preprocess inputs
    
    def preprocess(dx_type, age, sex, localization):

        df1 = pd.read_csv('Resources/df1.csv', nrows=0)  

        # Load the saved age scaler
        age_scaler = load('Model1/age_scaler.joblib')

        # Create a dictionary for input data
        input_data = {column: 0 for column in df1.columns.drop('result')}  # Exclude the target column

        # Label encode dx_type and diagnosis
        input_data['dx_type'] = dx_type_label_encoder.transform([dx_type])[0]

        # Set age
        input_data['age'] = age
        # Scale the age using the loaded scaler
        scaled_age = age_scaler.transform([[age]])[0][0]
        input_data['age'] = scaled_age

        
        # One-hot encode sex and localization
        input_data[f'sex_{sex}'] = 1
        input_data[f'localization_{localization}'] = 1

        
        # Convert the dictionary to a DataFrame
        input_df = pd.DataFrame([input_data])
        
        return input_df
        

    # Predict function
    def predict(dx_type, age, sex, localization):
        input_data = preprocess(dx_type, age, sex, localization)
        prediction = svm_model.predict(input_data)
        return prediction[0]
    

    # Predict Button
    if st.button('Predict'):
        result = predict(dx_type, age, sex, localization)
        if result == 0:
            st.success('This lesion has a possibility of being Benign.')
        else:
            st.error('This lesion has a possibility of being Malignant.')

    # Create a new row to display the additional text below the risk assessment tool
            
    additional_text_row = st.container()
    with additional_text_row:
        st.markdown("#### Discalimer")
        st.markdown(additional_text)

        

        

def image_to_base64(img):
    import base64
    from io import BytesIO

    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


# Run the Streamlit app
if __name__ == '__main__':
    app()    