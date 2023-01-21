import numpy as np
import joblib
import streamlit as st

# Lodaing the diabetes file.
model = joblib.load(open("C:/Users/om22s/Desktop/Author/Projects/Multiple-Disease-Classification/models/diabetes_model.joblib", 'rb'))

# Diabetes Prediction function :- 
def diabetes_prediction(input_data) :
    input_data_as_np_array = np.asarray(input_data)
    # reshape the array for prediction of one instance only.
    input_data_reshaped = input_data_as_np_array.reshape(1, -1)
    # prediction 
    return model.predict(input_data_reshaped)[0]
     
def main():
    # Title for the webpage.
    st.title("Diabetes Prediction Webapp")
    
    # Getting the input data from the user.
    Pregnancies = st.text_input("Number of pregnancies")
    Glucose = st.text_input("Glucose level")
    BloodPressure = st.text_input("Blood pressure level")
    SkinThickness = st.text_input("Skin thickness")
    Insulin = st.text_input("Insulin levels")
    BMI = st.text_input("BMI")
    DiabetesPedigreeFunction = st.text_input("Diabetes pedigree function")
    Age = st.text_input("Age")
    
    # Creating a button for prediction.
    if st.button("Diabetes Test Result") :
        st.write(diabetes_prediction([Pregnancies, Glucose, BloodPressure, 
                SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]))
    

if __name__ == "__main__" :
    main()
        
        
    


