import streamlit as st
import pandas as pd
from session import get_session,set_session,clear_session
import pymongo
st.set_page_config(
    page_title="Home",
    page_icon="👋",
    layout='centered'
)


import base64


@st.cache_data
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

img_file = 'streamlitapp/image1.jpg'
img_base64 = get_base64_of_bin_file(img_file)

page_bg_img = f'''
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/jpeg;base64,{img_base64}");
    background-size: cover;
}}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)
st.markdown(
    """
    <style>
    .header {
        font-size: 36px; /* Larger font size */
        font-weight: bold; /* Make text bold */
        color: #333333; /* Dark gray text color for better readability */
        background-color: #f8f9fa; /* Light background color */
        padding: 20px; /* Add padding */
        border-radius: 10px; /* Rounded corners */
        text-align: center; /* Center text */
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1); /* Subtle shadow for light theme */
        margin-bottom: 20px; /* Space below the header */
        text-transform: uppercase; /* Uppercase text */
        border: 1px solid #ddd; /* Light border to define edges */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header with the new style
st.markdown(
    """
    <div class="header">
        Welcome To Patient Adherence Analysis
    </div>
    """,
    unsafe_allow_html=True
)

# Define custom CSS for the button
st.markdown(
    """
    <style>
    .button-container {
        display: flex;
        justify-content: center;
        margin-top: 50px; /* Space from the top */
    }
    .button-container button {
        font-size: 18px;
        padding: 10px 20px;
        background-color: #007bff; /* Light blue background */
        color: #ffffff; /* White text color */
        border: none;
        border-radius: 5px;
        cursor: pointer;
        transition: background-color 0.3s; /* Smooth transition for hover effect */
    }
    .button-container button:hover {
        background-color: #0056b3; /* Darker blue on hover */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Center the button using HTML
st.markdown(
    """
    <div class="button-container">
        <button onclick="window.location.href='/pages/login.py'">Sign In</button>
    </div>
    """,
    unsafe_allow_html=True
)

def home_page():
    get_session()
    if st.session_state.get('logged_in'):
        st.write(f"Welcome, {st.session_state['username']}!")
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox("Choose a page", ["Main Page", "Logout"])
        
        if page == "Main Page":
            st.switch_page("pages/patient_details.py")  
        elif page == "Logout":
            clear_session()
            st.switch_page("pages/login.py") 
    else:
        st.sidebar.empty()

if __name__ == "__main__":
    home_page()
