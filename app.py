import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pickle
import json
import datetime as dt
from PIL import Image

##-----------------------------------------------------------------Pickle--------------------------------------------------------------------------------------------#
with open(r"D:\Singapore  Resale Flat Prices Predicting\random_sell.pkl",'rb') as file:
    price_model = pickle.load(file)
with open("D:\Singapore  Resale Flat Prices Predicting\Cat_Columns_Encoded_value.json",'rb') as file:
    encode_file = json.load(file)

st.set_page_config(page_title="Flat Resell Price Model", page_icon="D:\Singapore  Resale Flat Prices Predicting\icon.jpg",layout='wide')

# Set background image URL (you can replace this with your image URL)
background_image_url = "https://saltosystems.com/sites/default/files/styles/breakpoint_1920/public/images/contents/residential_background_1.jpg?itok=yErIXYOm"  

# Custom CSS to set background image for the entire page
st.markdown(f"""
    <style>
        /* Background and page styling */
        body {{
            background-image: url("{background_image_url}");
            background-size: cover;  /* Ensures the image covers the entire page */
            background-repeat: no-repeat;  /* Ensures the image does not repeat */
            background-attachment: fixed;  /* Makes the background fixed when scrolling */
            color: white;  /* Text color set to white for contrast */
        }}
        .stApp {{
            background-color: transparent;
        }}
        
        /* Option Menu Styling */
        .css-1v0mbp5 {{
            background-color: rgba(255, 255, 255, 0.7);  /* Semi-transparent background for option menu */
            border-radius: 10px;  /* Optional: round the corners */
        }}

        .header {{
            color: #FFC300 !important;
            font-size: 28px;  /* Increased font size */
            font-weight: bold;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);  /* Text shadow effect */
        }}

        /* Styling for content text with text-shadow and increased font size */
        .content-text {{
            color: white;
            font-size: 18px;  /* Increased font size */
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);  /* Light text shadow effect */
        }}
        
        .stColumn {{
            background: rgba(0, 0, 0, 0.3);  /* Semi-transparent black background */
            backdrop-filter: blur(10px);  /* Apply blur effect */
            border-radius: 10px;  /* Optional: round the corners */
            padding: 10px;
        }}

        .button-prompt {{
            color: #FFC300;  /* Custom yellow color */
            font-size: 18px;
            font-weight: bold;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
        }}

        /* Style the form submit button */
        .stForm button {{
            color: white;  /* Set text color */
            font-weight: bold;
            font-size: 18px;
            border-radius: 8px;
            padding: 15px 50px;  /* Make button wider */
            width: 50%;  /* Full width button */
            transition: background-color 0.3s ease;  /* Smooth transition for hover effect */
        }}

        /* Hover effect for the button */
        .stForm button:hover {{
            background-color: #FFC300;  /* Maintain the same color on hover */
            color: white;
        }}
    </style>
""", unsafe_allow_html=True)



st.markdown("<h1 style='text-align:center;text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5); color:#FFC300; font-size:50px;'>Singapore Resale Flat Prices Predicting Model</h1>", unsafe_allow_html=True)

#option menu

selected = option_menu(
    menu_title=None,
    options=["Home","Resell Price Prediction"],
    icons=["house", "bar-chart"],
    default_index=0,
    orientation="horizontal",
    styles={
        "nav-link-selected": {
            "background-color": "#FFC300",  # Highlighted option with Yellow
            "color": "white",  # White text when selected
        },
        "nav-link": {
            "color": "#ffffff",  # white for unselected options
        }
    }
)

if selected =='Home':
    col1,col2=st.columns(2)
    with col1:
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown('<h2 class="header">Overview</h2>', unsafe_allow_html=True)
        st.markdown('<p class="content-text">The objective of this project is to develop a machine learning model and deploy it as a user-friendly web application that predicts the resale prices of flats in Singapore. This predictive model will be based on historical data of resale flat transactions, and it aims to assist both potential buyers and sellers in estimating the resale value of a flat.</p>', unsafe_allow_html=True)
        st.markdown('<h2 class="header">Deliverables</h2>', unsafe_allow_html=True)
        st.markdown('<p class="content-text">The project will deliver the following deliverables:</p>', unsafe_allow_html=True)
        st.markdown('<p class="content-text">A well-trained machine learning model for resale price prediction A user-friendly web application (built with Streamlit). Documentation and instructions for using the application. A project report summarizing the data analysis, model development, and deployment process.</p>', unsafe_allow_html=True)
        st.markdown('<h2 class="header">Technologies Used</h2>', unsafe_allow_html=True)
        st.markdown('<p class="content-text">Python, Pandas, Numpy, Pickel, Matplotlib, Seaborn, Scikit-learn, Streamlit...</p>', unsafe_allow_html=True)
    with col2:
        st.image('https://fastercapital.co/i/Pipeline-regression--How-to-perform-regression-analysis-on-your-pipeline-data-and-outputs--Training-and-Evaluating-the-Regression-Model.webp')
        st.image('https://fastercapital.co/i/Random-forest--How-to-Use-Random-Forest-to-Improve-the-Performance-and-Accuracy-of-Your-Decision-Tree-Models--Advantages-of-Random-Forest.webp')

##----------------------------------------------------------Price prediction---------------------------------------------------------------------------------------------------------------------------#
if selected == 'Resell Price Prediction':

    st.markdown("<h1 style='text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5); color:#FFC300; font-size:35px;'>     Predicting price based on Trained data and Model</h1>", unsafe_allow_html=True)

    with st.form("Regression"):
        col1,col2=st.columns(2)
        with col1:
            start=1
            end=12
            month=st.number_input("select the **Transaction Month**",min_value=1,max_value=12,value=start,step=1)
            town=st.selectbox("Select the **Town**",encode_file['town_initial'])
            block=st.selectbox('Select the **Block**',encode_file['block_initial'])
            street=st.selectbox('Select the **Street**',encode_file['street_name_initial'])
            flat_type=st.selectbox('Select the **Flat Type**',encode_file['flat_type_initial'])
        
        with col2:
            year = st.number_input("Select the **Transaction Year**", min_value=1990, max_value=2025, value=dt.datetime.now().year)
            floor_area=st.number_input('Select the **Floor area**',value=28.0,min_value=28.0,max_value=307.0,step=1.0)
            flat_model=st.selectbox('Select the**Flat Model**',encode_file['flat_model_initial'])
            lease_year=st.number_input('Enter the **Lease Commence Year**', min_value=1966, max_value=2022, value=2017)
            storey_range = st.number_input('Select the **Storey Range**', value=0, min_value=0, max_value=100)
            

        with col1:
            st.markdown('<p class="button-prompt">Click below button to predict</p>', unsafe_allow_html=True)
            button=st.form_submit_button(label='Predict')
    
    if button:
        town_encode = encode_file['town_initial'].index(town)
        flat_type_encode = encode_file['flat_type_initial'].index(flat_type)
        block_encode = encode_file['block_initial'].index(block)
        street_name_encode = encode_file['street_name_initial'].index(street)
        flat_model_encode = encode_file['flat_model_initial'].index(flat_model)

        input_ar = np.array([[town_encode,flat_type_encode,block_encode,street_name_encode,storey_range,floor_area, flat_model_encode,lease_year,year,month,]],dtype=np.float32)
        Y_pred=price_model.predict(input_ar)
        sell_price=round(Y_pred[0],2)

        with col2:
            st.markdown("")
            st.markdown(f'<h2 class="header">Predicted Resell Price is: {sell_price}</h2>', unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")