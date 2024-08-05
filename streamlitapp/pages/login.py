import streamlit as st
from auth import login_user
from session import clear_session


def login_page():
    st.title("Login")
    if 'logged_in' in st.session_state and st.session_state['logged_in']:
        st.switch_page("pages/patient_details.py")
    with st.form(key='login_form'):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login")
        
        if submit_button and login_user(email,password):
            st.success("Login successful!")
            st.switch_page("pages/patient_details.py")  # Switch to the home page
        else:
            if email and password:
                st.error("Invalid username or password")
           

if __name__ == "__main__":
    login_page()
