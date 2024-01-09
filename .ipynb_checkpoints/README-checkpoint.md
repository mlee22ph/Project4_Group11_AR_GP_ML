# Skin Cancer Prediction and Classification

Project4 - UWA/edX Data Analytics Bootcamp

Github repository at: [https://github.com/mlee22ph/Project4_Group11_AR_GP_ML.git](https://github.com/mlee22ph/Project4_Group11_AR_GP_ML.git)

## Table of Contents

1. [Introduction](#introduction)
   1. [Overview](#overview)
   2. [Objective](#objective)
2. [Getting Started](#getting-started)
   1. [Prerequisites](#prerequisites)
   2. [Installation](#installation)
   3. [Running the Application](#running-the-application)
3. [Features](#features)
4. [Approach](#approach)
   1. [Methodology](#methodology)
   2. [Structure](#structure)
   3. [Scripts](#scripts)
5. [Repository Structure](#repository-structure)
6. [Data Sources and Copyright](#data-sources-and-copyright)
   1. [Data Sources](#data-sources)
   2. [Copyright Notice](#copyright-notice)
7. [Conclusion](#conclusion)
8. [References](#references)


## Introduction

### Overview

The Skin Cancer Prediction offer an early screening using a few parameters (age, sex, diagnostic type and body location). Skin Cancer Classification is also done using data from the available extracted RGB file which was derived from the 10015 images.  Classification using the images directly was also attempted.

This is a team effort with the following contributors:
- Athira R
- Mike L
- Geoffrey P

### Objective

Our objective is to provide early screening based on a few parameters and image to give an indication whether the person has skin cancer or not and what type of skin cancer.

## Getting Started

### Prerequisites

- Python (version 3.x recommended)
- streamlit (Do !pip install streamlit if not done yet)
- Install TensorFlow or use Google Colab to run Model3 jobs due to the large image size.
- A modern web browser

### Installation

To get started with the dashboard:

1. Clone the repository to your local machine:
   ```
   git clone https://github.com/mlee22ph/Project4_Group11_AR_GP_ML.git
   ```
2. (Optional) Set up a virtual environment in the project directory.
3. Install the required dependencies:
   ```
   !pip install streamlit
   !pip install kaggle
   !pip install joblib
   ```

### Running the Application

1. Open your terminal and navigate to the project directory.
2. Activate the virtual environment, if you have set one up.
3. Start the streamlit application:
   ```
   streamlit run streamlit_app2.py
   ```
4. The Skin Cancer Prediction & Classification App will automatically pop-in in Chrome


## Features

- **Select Diagnosis Type**: Users can choose from a dropdown menu to select a diagnosis type.
- **Enter Age**: User to type their age from 0 to 100 (integer).
- **Select Sex**: Users can choose from a dropdown menu to select Sex.
- **Select Localization**: Users can choose from a dropdown menu to select Location.
- **Predict Button**: Users to click this button once all entries are checked.  This will produce a result of "Benign" or "Cancerous" based on the input after running through the trained model.

## Approach

### Methodology
- The project use the CRUD in data ETL, select the prediction and associated features.  Split the data to train and test, apply scaling, build model and make prediction.  Final step is model evaluation and further iteration to improve model prediction accuracy.

**Data Acquisition and Analysis:**

1. **Data Downloading:** We are able to get all the necessary files from an existing kaggle project.  https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
2. **The key Files downloaded are:** 
    - HAM10000_images_part1.zip - this contain 5000 images
    - HAM10000_images_part2.zip - this contain 5015 images
    - HAM10000_metadata.csv - this contain key metadata for the 10015 cases
    - hmnist_28_28_RGB.csv - this contain the RGB data per image in 28 by 28 pixels

**Data Processing and Database Creation:**

3. **ETL Process:** The data gathered underwent an Extract, Transform, Load (ETL) process, primarily using Python's pandas library.
- Refer to project schema diagram. [!Project Schema](./Images/ASX_top_ten_ERD.jpg)](https://github.com/mlee22ph/Project4_Group11_AR_GP_ML.git/blob/main/Images/ASX_top_ten_ERD.jpg)

**Application Development and Deployment:**

4. **Front-End Development:** The dashboard's front end was created using Streamlit.


### Structure

- **Frontend:** Steamlit application


## Scripts 

- ETL Directory: Contains notebook used for ETL 
	- `ETL.ipynb` - Jupyter notebook for extract, transform and load the data
- Model 1 Directory: Contains all notebooks for skin cancer prediction using DecisionTree, LogisticRegression, RandomForest andd SVM, including saved models and encoders.
	- `SkinCancerClasification.ipynb` - Jupyter notebook using NN model without scaling
- Model 2 Directory: Contains all notebooks for skin cancer classification, including saved models and images
	- `SkinCancerClasification.ipynb` - Jupyter notebook using NN model without scaling
	- `SkinCancerClasificationNoOverSampler.ipynb` - Jupyter notebook using NN model without scaling and without OverSampler
	- `SkinCancerClasificationNoOverSamplerOptimised.ipynb` - Jupyter notebook using NN model without scaling and without OverSampler but with optimised parameters to improve accuracy
	- `SkinCancerClasificationNoOverSamplerWithScaling.ipynb` - Jupyter notebook using NN model with scaling but without OverSampler 
	- `SkinCancerClasificationNoOverSamplerWithScalingOptimised.ipynb` - Jupyter notebook using NN model with scaling but without OverSampler and with optimised parameters to improve accuracy
	- `SkinCancerClasificationWithScaling.ipynb` - Jupyter notebook using NN model with scaling
	- `SkinCancerClasificationWithScaling1stOptimisation.ipynb`  - Jupyter notebook using NN model with scaling and with OverSampler and the initial optimisation to improve accuracy
	- `SkinCancerClasificationWithScalingFinalOptimisation.ipynb`  - Jupyter notebook using NN model with scaling and with OverSampler and the final optimisation to improve accuracy
	- `SkinCancerClasificationWithScalingMoreOptimisations.ipynb`  - Jupyter notebook using NN model with scaling and with OverSampler and the second to the last optimisation to improve accuracy
	- Models Directory containing all saved models for each notebook above
	- Images Directory containing all saved images for each notebook above
- Model 3 Directory: Contains all notebooks for skin cancer image classification, including saved associated models and parameters. Please note that these notebooks were run in Google Colab and saved and copied here.  The directory path in the notebooks will need to be changed to make it work.
	- `Model3_28_10000.ipynb` - Jupyter notebook using NN model using 10000 images with 28 by 28 pixel
	- `Model3_28_350.ipynb` - Jupyter notebook using NN model using 350 images with 28 by 28 pixel
- Resource Directory: Contains all data input and subsequent output.
	- `HAM10000_metadata.csv` - metadata csv file downloaded from the kaggle.com
	- `hmnist_28_28_RGB.csv` - RGB in 28x28 pixels csv file downloaded from the kaggle.com
	- various other output and processed csv
	- `test_1_image_per_dx` - folder containing 7 images one each per dx column.  Images used as input in final model saved.
	- `train_50_images_per_dx` - folder containing 50 images each per dx column.  Used for the `Model3_28_350.ipynb` job.
- Streamlit Directory: Contains all notebooks for streamlit app deployment with the following subfolders and file.
	- streamlit_app2.py - streamlit code
    


## Repository Structure

- **Root Directory:** Readme and the main folders.
- **ETL Directory:**
- **Model1 Directory:** Contains all the Skin Cancer Classification notebooksand other files
- **Model2 Directory:**
- **Model3 Directory:** 
- **Streamlit Directory:** Contains all notebooks for streamlit app deployment with the following subfolders and file.
	- Images - skin_cancer.jpg for display
	- Model1 - best model1 for deployment with parameter scaling and model saved.
	- Resources - csv files
- **Resources Directory:** Contains all data input and subsequent output.
	- `HAM10000_metadata.csv` - metadata csv file downloaded from the kaggle.com
	- `hmnist_28_28_RGB.csv` - RGB in 28x28 pixels csv file downloaded from the kaggle.com
	- various other output and processed csv
	- `test_1_image_per_dx` - folder containing 7 images one each per dx column.  Images used as input in final model saved.
	- `train_50_images_per_dx` - folder containing 50 images each per dx column.  Used for the `Model3_28_350.ipynb` job.

## Data Sources and Copyright

### Data Sources

The data used in this dashboard is sourced from the following:

- **kaggle.com**: The dataset was taken from kaggle.com but was originally taken from Harvard Unversity.
(https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T).


### Copyright Notice

This application has been created to fulfill the requirements of the Data Analytics Boot Camp hosted by UWA in 2023 and should not be interpreted as medical advice. 
    Working with a limited dataset, time and expertise the information provided by this app may not be entirely accurate, as the primary intention was to implement and demonstrate the skills learned during the course. The focus was more on skill application rather than ensuring the accuracy of the data predictions. 
    This project is primarily meant for exploring and showcasing the student's knowledge, rather than providing reliable medical analysis or advice. rediciton is

## Conclusion

This Skin Cancer Prediction and Classification streamlit is able to offer early screening for skin cancer prediction based on four criteria.  Classification using the RGB dataset produce robust result but the classification using images has not been able to yield reliable prediction and need further work to improve.

## References

- Codes and approaches inspired by lecture notes and ChatGPT.



