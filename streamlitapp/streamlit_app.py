import streamlit as st
import pandas as pd
from session import get_session,set_session,clear_session
import pymongo
st.set_page_config(
    page_title="Home",
    page_icon="👋",
    layout='centered'
)

st.header("Welcome To patient adherence analysis")
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
