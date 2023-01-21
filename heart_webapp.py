import streamlit as st
import numpy as np
import joblib
# Lodaing the diabetes file.
model = joblib.load(open("C:/Users/om22s/Desktop/Author/Projects/Multiple-Disease-Classification/models/heart_model.joblib", 'rb'))

# Diabetes Prediction function :- 
def heart_prediction(input_data) :
    input_data_as_np_array = np.asarray(input_data)
    # reshape the array for prediction of one instance only.
    input_data_reshaped = input_data_as_np_array.reshape(1, -1)
    # prediction 
    return model.predict(input_data_reshaped)[0]
     

def main():
    # Title for the webpage.
    st.title("Heart Disease Prediction Webapp")
    
    # Getting the input data from the user.
    Age = st.text_input("Age")
    Sex = st.text_input("Sex")
    Cp = st.text_input("Cp")
    Trestbps = st.text_input("Trestbps")
    Chol = st.text_input("Cholestrol")
    Fbs = st.text_input("Fbs")
    RestECG = st.text_input("RestECG")
    Thalach = st.text_input("Thalach")
    Exang = st.text_input("Exang")
    Oldpeak = st.text_input("Oldpeak")
    Slope = st.text_input("Slope")
    Ca = st.text_input("Ca")
    Thal = st.text_input("Thal")
    
    # Creating a button for prediction.
    if st.button("Heart-Disease Test Result") :
        diagnosis = heart_prediction([Age, Sex, Cp, Trestbps, Chol,Fbs, 
                                      RestECG, Thalach, Exang, Oldpeak, 
                                      Slope, Ca, Thal])
        st.write(diagnosis)

if __name__ == "__main__" :
    main()
