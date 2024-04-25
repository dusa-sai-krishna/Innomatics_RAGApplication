import streamlit as st
from utils import getResponse

st.set_page_config(
    page_title='RAG Application',
    page_icon='📝'
)

st.title('RAG Application')

question=st.text_area('Enter your question')
submit=st.button('Submit')
if submit==True:
    
    st.write(getResponse(question))