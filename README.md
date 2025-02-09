## Singapore-Flat-Resale_Price
## Introdction:
The objective of this project is to develop a machine learning model and deploy it as a user-friendly web application that predicts the resale prices of flats in Singapore. This predictive model will be based on historical data of resale flat transactions, and it aims to assist both potential buyers and sellers in estimating the resale value of a flat.

## Technologies used:
Data Collection, Data Wrangling, Data Preprocessing, EDA, Model Building, Model Deployment.

## Libraries Used:

* Streamlit
* Pandas
* Numpy 
* Random Forest Regression
* Json

## Flow of the Project:

* Download the dataset from the link https://beta.data.gov.sg/collections/189/view
* Totally there are 5 datasets ranging from 1990 to 2022
* Download the complete datasets, did data wrangling and concate it to a one dataset
* Data preprocessing steps includes fill null values, datatype conversion, outliers detection etc.,
* Created additional features based on the datasets
* Visualizations done for categorical and continuous variables
* Visualizations like bar chart, count plot, box plot, line plot
* Label encoding was used to encode categorical variables
* Highly correlated features with target variables was calculated using correlation matrix and heat map
* Split the dataset into train and test set for model building
* Used Linear regression and Decision tree regression for model building
* Coefficient of determination for Linear regression is 0.77 and that for Decisiontree regression is 0.97
* Decision Tree was used to build predictive model
* Using pickling , pickle the model for further usage
* Created a simple streamlit  UI for the users to set the values for predicting resale flat price in Singapore
* Price per square meter, Flat type, Floor area square meter are the few features highly impacting the resale price
* Successfully deployed the predictive modelling in Render platform, to use this predictive modelling as a  streamlit web application.
* Json
