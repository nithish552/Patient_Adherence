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

img_file = 'pages\image1.jpg'
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

st.header("Welcome To")
st.header("Patient Adherence Analysis")
if st.button(label="Sign in"):
    st.switch_page('pages/login.py')


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
