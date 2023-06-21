import streamlit as st
import pandas as pd
import numpy as np
from prediction import PREDICT


st.title("ML MODEL:green[: Predicting if a passenger would Survive or not in TITANIC CRUISE]")

st.header("Please choose Gender and Ticket Class")
col1, col2 = st.columns(2)

with col1:
    
    Passenger_gender = st.radio("** Please Choose gender **", options = ["Male", "Female"])
    if Passenger_gender == "Male":
        Passenger_gender = 1
    else:
        Passenger_gender = 0
    
with col2:
    
    Ticket_class= st.radio("** Please Choose Ticket class **", options = ["1: First Class","2: Second Class","3: Third Class"])
    if Ticket_class == "1: First Class":
        Ticket_class = 1
    elif Ticket_class == "2: Second Class":
        Ticket_class = 2
    else:
        Ticket_class = 3

st.text("NOTE: The accuracy of the prediciton is 79.85%")
if st.button("predict Survives or not"):
    result = PREDICT(np.array([[Ticket_class,Passenger_gender]]))
    #st.text(result[0])
    
    # Check if predictions is not None before accessing its elements
    if result is not None:
        if result[0] == 1:
            st.markdown("**PREDICTION( accuracy 79.85% ): The passenger :green[SURVIVES]**")
            
        else:
            st.markdown("**PREDICTION( accuracy 79.85% ): The passenger will :red[not SURVIVE]**")
    else:
        st.text("No predictions available")