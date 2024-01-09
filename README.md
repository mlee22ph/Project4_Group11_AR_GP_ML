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

Our objective is to provide early screening based on a few parameters and image to given an indication whether the person has skin cancer or not and what type of skin cancer.

## Getting Started

### Prerequisites

- Python (version 3.x recommended)
- streamlit (Do !pip install streamlit if not done yet)
- Install TensorFlow or use Google Colab
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
- Refer to Project Workflow diagram.  [!Project Workflow](./Images/Project_Workflow.jpg)](https://github.com/mlee22ph/Project4_Group11_AR_GP_ML.git/blob/main/Images/Project_Workflow.jpg)

**Data Acquisition and Analysis:**

1. **Data Downloading:** We are able to get all the necessary files from an existing kaggle project.  https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
2. **The key Files downloaded are:** 
    - HAM10000_images_part1.zip - this contain 5000 images
    - HAM10000_images_part2.zip - this contain 5015 images
    - HAM10000_metadata.csv - this contain key metadata for the 10015 cases
    - hmnist_28_28_RGB.csv - this contain the RGB data per image in 28 by 28 pixels

**Data Processing and Database Creation:**

5. **ETL Process:** The data gathered underwent an Extract, Transform, Load (ETL) process, primarily using Python's pandas library.
- Refer to project schema diagram. [!Project Schema](./Images/ASX_top_ten_ERD.jpg)](https://github.com/mlee22ph/Project4_Group11_AR_GP_ML.git/blob/main/Images/ASX_top_ten_ERD.jpg)

**Application Development and Deployment:**

7. **API Development:** A Flask application was developed to generate APIs. These APIs allow for various queries, either targeting specific tables or joining them to fetch necessary data for our dashboard.
8. **Front-End Development:** The dashboard's front end was created using Streamlit.


### Structure

- **Frontend:** Steamlit application


## Scripts 

- `01_ASX_Top10_Dataframes_Historic.ipynb`: Jupyter notebook for retrieving top 10 companies per industry group and historical stock data.
- `02_Retrive_ASX_Ticker_Fundamentals.ipynb`: Script for scraping fundamental data of ASX-listed companies.
	- Total of 250 tickers were scraped.  Scraping done 10 tickers at a time and later merged to create the raw CSV file.
	- various steps were taken to clean the dataset to create the final clean CSV file.
	- `PE` column was computed using the formula `PE = lastPrice/EPS` and ensuring the value is zero if EPS is zero or Null.
- `03_Creating_DataBase.ipynb`: Notebook detailing the creation of the database structure.
	- 4 tables were created `industry_groups`, `top_ten`, `top_ten_historic` and `fundamental` in the `top_ten_asx.db` database. 
- `app_solution.py`: The Flask application script.
	- 4 json end points are created corresponding to the 4 tables.
- `index.html`: Main HTML file that structures the web dashboard.
- `plot.js`: script for fetching the neccesary data, plotting interactive charts, updating tables and initialising the dashboard.
	- the url variables corresponding to the 4 json end points are declared.
	- `fetchData()` - an asynchronous function design to fetchdata from url.
	- `populateIndustryGroupDropdown()` - function designed to populate the industryGroups drowndown.
	- `populateTickerDropdown()` - function designed to populate the ticker drowndown based on industryGroup selected.
	- `updateStockInfo()` - function designed to populate the Stock Information per ticker.
	- `updateTimeSeriesChart()` - function designed to create line chart per ticker
	- `updateBarCharts()` - function designed to create bar charts for `Market Cap`, `EPS`, `DPS` and `PE`.
	- `optionChanged()` - function to handle the change in dropdown value
	- `initializePage()` - function to initialize the page
- `styles.css` : script for formatting the dashboard

## Repository Structure

- **Root Directory:** Readme and the main folders.
- **ETL Directory:**
- **Model1 Directory:** Contains all the Classification notebooks including an Image folder containing all images resulting from the notebooks
- **Model2 Directory:**
- **Model3 Directory:**
- **Streamlit Directory:** Includes assets like `ASX_top_ten_ERD.jpg`.
- **Resources Directory:** Contains datasets in csv format.

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

This Skin Cancer Prediction and Classification streamlit is able to offer early screeening for skin cancer prediction based on four criteria.  The classification using using images has not been able to yield reliable prediction and need further work to improve.

## References

- Codes and approaches inspired by lecture notes and ChatGPT.



