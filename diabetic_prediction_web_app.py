# -*- coding: utf-8 -*-
"""
Created on Sat Sep 20 19:09:38 2025

@author: khana
"""

import numpy as np
import pickle 
import streamlit as st
#loading the saved model
loaded_model=pickle.load(open('C:/Users/khana/Downloads/trained_model.sav','rb'))

#function for prediction
def diabetes_prediction(input_data):
    #creating numpy array
    ip_as_np=np.asarray(input_data)
    input_data_reshaped=ip_as_np.reshape(1,-1)
    prediction=loaded_model.predict(input_data_reshaped)
    print(prediction)
    if (prediction[0]==0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'
  
    


def main():
    #title for the webpage
    st.title('Diabetes Prediction Web App')
    #we need 8 input to give out prediction
    Pregnancies=st.text_input('Number of pregnancies')
    Glucose=st.text_input('Glucose Level')
    BloodPressure=st.text_input('Blood pressure value')
    SkinThickness=st.text_input('Skin thickness value')
    Insulin=st.text_input('Insulin value')
    BMI=st.text_input('BMI value')
    DiabetesPedigreeFunction=st.text_input('Diabetes pedigree function value')
    Age=st.text_input('Age')
    
    #to store the string output from the diabetes_prediction function
    diagnosis=''
    
    #whenever the button is pressed the diabetes_prediction function will be called and string returned will be stored in variable
    if st.button('Diabetes test result'):
        diagnosis=diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
     
    #to give out the final output or to print the output   
    st.success(diagnosis)
    
    
    
#to ensure main function will only run after the anaconda command promt
#this block ensures that main() is only executed when the script is run directly, not when imported.
if __name__=='__main__':
    main()
    
    