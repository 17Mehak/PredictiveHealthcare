# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 14:09:15 2023

@author: Mehak Jain
"""

import pickle 
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import matplotlib.pyplot as plt
# loading the saved model 

img = Image.open(r"C:\Users\Mehak Jain\Desktop\Disease Prediction System\Home1.jpg")
img1 = Image.open(r"C:\Users\Mehak Jain\Desktop\Disease Prediction System\heart.jpg")
img2 = Image.open(r"C:\Users\Mehak Jain\Desktop\Disease Prediction System\kidney.jpg")
img3 = Image.open(r"C:\Users\Mehak Jain\Desktop\Disease Prediction System\liver.jpg")
img4 = Image.open(r"C:\Users\Mehak Jain\Desktop\Disease Prediction System\diabetes.jpg")
img5 = Image.open(r"C:\Users\Mehak Jain\Desktop\Disease Prediction System\Home2.png")


heart_model = pickle.load(open('C:/Users/Mehak Jain/Desktop/Disease Prediction System/heart_model (2).sav', 'rb'))
diabetes_model = pickle.load(open('C:/Users/Mehak Jain/Desktop/Disease Prediction System/diab_model.sav', 'rb'))
liver_model = pickle.load(open('C:/Users/Mehak Jain/Desktop/Disease Prediction System/liv_model.sav', 'rb'))
kidney_model = pickle.load(open('C:/Users/Mehak Jain/Desktop/Disease Prediction System/kidney_model (1).sav', 'rb'))
# sidebar 


with st.sidebar: 
    
    selected = option_menu(
        menu_title = 'Multiple Disease Prediction using Machine Learning',
        options = ['Home','Our Models','Diabetes Prediction','Heart Disease Prediction','Liver Disease Prediction','Kidney Disease Prediction'],
                           default_index=0)




    

    
if (selected == 'Home'):
    
    #col1, col2 = st.columns(2)
    
    
    #with col1: 
    #st.write("**Healthcare with AI**")
        
   # with col2:

    st.markdown("<h1 style='text-align: center; color: black;'>Healthcare Prediction Using Machine Learning</h1>", unsafe_allow_html=True)
    
    st.markdown("""
                <style>
                .center-image {
                    display: block;
                    margin: 0 auto;
                    }
                </style>
                """, unsafe_allow_html=True)

# Display the image with the CSS class applied
   # st.markdown("<img src='Home.jpg' class='center-image'>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

# Display the image in the second column
    
    with col1:
        st.image(
            img5,
            width = None
            )
    with col2:
        st.image(
            img,
            width = None
            )
    
    
    #col1, col2 = st.columns(2)
   
    #with col1:
    st.header("The Power of Early Detection: Transforming Lives Through Machine Learning")
    st.write("In the realm of healthcare, early detection has emerged as a game-changer, significantly impacting the lives of millions. Detecting diseases at their nascent stages not only improves treatment outcomes but also enhances the overall quality of life for individuals. Here's why early detection is so crucial and how Machine Learning is revolutionizing this essential aspect of healthcare.")
    st.write("**1. Timely Intervention, Improved Outcomes:** Early detection allows for timely intervention, often when the disease is still localized and hasn’t spread significantly. This enables doctors to prescribe more effective treatments, increasing the likelihood of successful outcomes. Whether it's cancer, diabetes, cardiovascular issues, or any other ailment, catching it early can make a substantial difference in treatment success rates.")
    st.write("**2. Cost-Effective Healthcare:** Early detection not only saves lives but also reduces the economic burden on individuals and healthcare systems. Preventive measures and less aggressive treatments at the early stages of diseases are generally more cost-effective, leading to significant savings in medical expenses in the long run.")
    st.write("**3. Enhanced Quality of Life:** For many chronic conditions, early detection means managing the disease effectively, slowing down its progression, and minimizing complications. This translates to a higher quality of life for individuals. Early intervention often allows individuals to continue their daily activities and maintain a sense of normalcy in their lives.")
    st.write("**4. Precision Medicine with Machine Learning:** Machine Learning, a subset of Artificial Intelligence, has emerged as a powerful tool in early disease detection. By analyzing vast amounts of patient data, ML algorithms can identify subtle patterns and markers associated with specific diseases. This analytical prowess enables healthcare professionals to make more accurate predictions and diagnosis, ensuring that interventions occur at the right time.")
    #with col2: 
    # st.header("Benefits of Disease Prediction")
    st.write("**5. Continuous Improvement:** One of the remarkable features of Machine Learning is its ability to learn and adapt continuously. As more data becomes available, ML algorithms become more accurate and sophisticated in their predictions. This continuous improvement ensures that healthcare practices are always evolving, leading to better outcomes for patients.")
    st.write("In essence, early detection, powered by the capabilities of Machine Learning, is reshaping the landscape of healthcare. It's not merely about identifying diseases; it's about offering hope, improving lives, and building a future where diseases are not just treated but are prevented effectively. Embracing the potential of early detection through Machine Learning is a step towards a healthier, happier world.")
   
    
    #st.header("Our models")
    #st.button("Heart Disease")
        
st.markdown(
    """
<style>
button {
    height: auto;
    padding-top: 10px !important;
    padding-bottom: 10px !important;
}
</style>
""",
    unsafe_allow_html=True,
)


if (selected == 'Our Models'):
   
   st.markdown("<h1 style='text-align: center; color: black;'>Our Models</h1>", unsafe_allow_html=True)
   
   
   # Diabetes 
   st.image(
           img4,
           width = 700
           
           )
   st.markdown("<h3 style='text-align: center; color: black;'>Diabetes Risk Assessment Model</h3>", unsafe_allow_html=True)   
   st.write("Our Diabetes Risk Assessment Model is a state-of-the-art tool developed to evaluate an individual's risk of developing diabetes. This model takes into account various risk factors, including family history, body mass index (BMI), age, and lifestyle choices. By utilizing advanced machine learning algorithms, it provides an accurate assessment of the likelihood of developing diabetes. Early identification of diabetes risk is crucial for proactive preventive measures and promoting a healthier lifestyle.")
   # Get accuracy (you need to replace this with your actual accuracy value)
   accuracy = 0.804

   # Display accuracy using a bar chart
   st.write("**Accuracy: 80.4%**")
   fig, ax = plt.subplots(figsize=(6, 3))
   ax.barh(['Accuracy'], [accuracy], color='green')
   ax.set_xlim(0, 1)
   ax.set_xlabel('Accuracy')
   ax.set_title('Accuracy Visualization')
   st.pyplot(fig)

   # Heart    
   
   st.image(
          img1,
          width = 700
          
          )
   st.markdown("<h3 style='text-align: center; color: black;'>Heart Disease Risk Assessment Model  </h3>", unsafe_allow_html=True)
   st.write("Our Heart Disease Risk Assessment Model is engineered to evaluate the probability of cardiovascular diseases. It considers vital factors such as blood pressure, cholesterol levels, family medical history, and lifestyle choices. Using sophisticated algorithms, the model calculates the risk score and provides insights into potential heart-related issues. Early detection facilitated by this model can significantly improve patient outcomes.")       
   # Get accuracy (you need to replace this with your actual accuracy value)

# Get accuracy (you need to replace this with your actual accuracy value)
   accuracy = 0.905

# Display accuracy using a bar chart
   st.write("**Accuracy: 90.5%**")
   fig, ax = plt.subplots(figsize=(6, 3))
   ax.barh(['Accuracy'], [accuracy], color='green')
   ax.set_xlim(0, 1)
   ax.set_xlabel('Accuracy')
   ax.set_title('Accuracy Visualization')
   st.pyplot(fig)
   
   
   # Kidney 
   
   st.image(
           img2,
           width = 700
        )
   st.markdown("<h3 style='text-align: center; color: black;'>Kidney Disease Detection Model </h3>", unsafe_allow_html=True)
   st.write("Our Kidney Disease Detection Model is a powerful tool in diagnosing renal problems. It utilizes a diverse range of patient data, including serum creatinine levels, glomerular filtration rate, and urine albumin levels. By applying advanced statistical methods, this model can identify kidney diseases in their early stages, allowing healthcare professionals to implement appropriate treatments and lifestyle changes promptly.")
   # Get accuracy (you need to replace this with your actual accuracy value)
   accuracy = 0.971

   # Display accuracy using a bar chart
   st.write("**Accuracy: 97.1%**")
   fig, ax = plt.subplots(figsize=(6, 3))
   ax.barh(['Accuracy'], [accuracy], color='green')
   ax.set_xlim(0, 1)
   ax.set_xlabel('Accuracy')
   ax.set_title('Accuracy Visualization')
   st.pyplot(fig)

  
   # Liver
   
   st.image(
            img3,
            width = 700
            )
   st.markdown("<h3 style='text-align: center; color: black;'>Liver Disease Prediction Model </h3>", unsafe_allow_html=True)
   st.write("Our Liver Disease Prediction Model is a cutting-edge tool designed to assess the risk of liver diseases. It analyzes a comprehensive set of patient data, including liver enzyme levels, medical history, and lifestyle factors. By employing advanced machine learning algorithms, this model accurately predicts the likelihood of liver diseases, enabling early diagnosis and timely intervention.")
   # Get accuracy (you need to replace this with your actual accuracy value)
   accuracy = 0.845

   # Display accuracy using a bar chart
   st.write("**Accuracy: 84.5**")
   fig, ax = plt.subplots(figsize=(6, 3))
   ax.barh(['Accuracy'], [accuracy], color='green')
   ax.set_xlim(0, 1)
   ax.set_xlabel('Accuracy')
   ax.set_title('Accuracy Visualization')
   st.pyplot(fig)

   
   
   
# Heart Disease Prediction Page
pain_type = ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"]
if (selected == 'Heart Disease Prediction'):
   st.title('Heart Disease Prediction using ML')
   st.write("Welcome to our Heart Disease Prediction page! Heart disease is a significant health concern worldwide, and early prediction plays a crucial role in prevention and timely medical intervention. Our advanced Machine Learning model can help assess the risk of heart disease based on various factors. Let's explore how it works and how you can use it for your health assessment.")
   st.write("**Understanding Heart Disease**")
   st.write("**What is Heart Disease?**")
   st.write("Heart disease, also known as cardiovascular disease, encompasses a range of conditions that affect the heart. It is a leading cause of death globally, and timely diagnosis is essential for effective treatment.")
   st.write("**How Machine Learning Helps**")
   st.write("Machine Learning algorithms analyze multiple factors such as age, blood pressure, cholesterol levels, and lifestyle habits to identify patterns associated with heart disease. By processing these factors, our model predicts the likelihood of an individual developing heart disease.")
  
   st.header("How to Use Our Heart Disease Prediction Tool")
   st.write("To use our prediction tool, simply enter the required information in the form below, and our Machine Learning model will provide you with an assessment of your heart disease risk.")
   st.write("1. **Age:** Enter your age in years.")
   st.write("2. **Sex:** Select your gender (Male/Female).")
   st.write("3. **Chest Pain Type:** Describe your chest pain type.")
   st.write("4. **Resting Blood Pressure:** Enter your resting blood pressure in mm Hg.")
   st.write("5. **Cholesterol Level:** Enter your cholesterol level in mg/dl.")
   st.write("6. **Fasting Blood Sugar:** Select whether your Fasting Blood Sugar Level is greater than 120mg/dl (Yes/No).")
   st.write("7. **Resting Electrocardiographic Results:** Describe your resting electrocardiographic results.")
   st.write("8. **Maximum Heart Rate Achieved:** Enter your maximum heart rate achieved.")
   st.write("9. **Exercise Induced Angina:** Select whether you experience exercise-induced angina (Yes/No).")
   st.write("10. **ST Depression Induced by Exercise:** Enter the ST depression induced by exercise.")
   st.write("11. **Slope of the Peak Exercise ST Segment:** Describe the slope of the peak exercise ST segment.")
   st.write("12. **Number of Major Vessels Colored by Flourosopy:** Enter the number of major vessels colored by flourosopy (0-3).")
   st.write("13. **Thalassemia Type:** Describe your thalassemia type.")
   st.header("Enter the following parameters: ")
    
 
   col1, col2, col3 = st.columns(3)
    
   with col1:
        age = st.text_input('Age')
        
   with col2:
        sex = st.selectbox("Select Gender", ["Female", "Male"])
        
   with col3:
        pain_types = st.selectbox("Select Chest Pain Type", pain_type)
        
        cp = -1
        if pain_types == "Typical Angina":
            cp = 0
        elif pain_types == "Atypical Angina":
            cp = 1
        elif pain_types == "Non-anginal pain":
            cp = 2
        else:
            cp = 3
        
        
        
   with col1:
        trestbps = st.text_input('Resting Blood Pressure')
        
   with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
        
   with col3:
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl',["Yes", "No"])
        
   with col1:
        restecg1 = st.selectbox('Resting Electrocardiographic results',["Normal", "Abnormal", "Unknown"])
        
        restecg = -1
        if restecg1 == "Normal":
            restecg = 0
        elif restecg1 == "Abnormal":
            restecg = 1
        else: 
            restecg = 2
        
        
        
   with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
        
   with col3:
        exang = st.selectbox('Exercise Induced Angina (Yes/No)', ["Yes", "No"])
        
   with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
        
   with col2:
        slope1 = st.selectbox('Slope of the peak exercise ST segment',["Upsloping", "Flat", "Downsloping"])
        slope = -1
        if pain_types == "Upsloping":
            slope = 0
        elif pain_types == "Flat":
            slope = 1
        else:
            cp = 3
   with col3:
        ca = st.text_input('Major vessels colored by flourosopy')
        
   with col1:
        thal1 = st.selectbox('Thalassemia Type', ["Normal","Fixed Defect", "Reversable Defect"])
        thal = -1
        if thal1 == "Normal":
            thal = 0
        elif thal1 == "Fixed Defect":
            thal = 1
        else:
            thal = 2
        
        
   sex = 1 if sex == "Male" else 0
  # cp = 0 if cp == "Typical Angina" else cp = 1 if cp == "Atypical Angina" cp = 2 if cp == "Non-Anginal Pain" else 3
   fbs = 1 if fbs == "Yes" else 0
   exang = 1 if exang == "Yes" else 0
     
    # code for Prediction
   heart_diagnosis = ''
   #probability = ''
   input_data = (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
   input_data_as_numpy_array= np.asarray(input_data)
   input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    
    # creating a button for Prediction
   if st.button('Heart Disease Test Result'):
        #heart_prediction = heart_model.predict([[age, sex, selected_chest_pain_numeric, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        heart_prediction = heart_model.predict(input_data_reshaped)
        heart_probability = heart_model.predict_proba(input_data_reshaped)[0][1]
        if heart_prediction[0] == 1:
            st.write("It's crucial to prioritize heart health. Schedule an appointment with a cardiologist for a detailed evaluation. Lifestyle changes, such as regular exercise, a heart-healthy diet, managing stress, and quitting smoking, can significantly reduce the risk of heart-related issues. Your healthcare provider will guide you on suitable medications and preventive measures.")
            heart_diagnosis = 'Heart Disease Risk Identified'
            
        else:
            heart_diagnosis = 'Heart Disease Risk Not Identified'
            st.write("A negative prediction for heart disease is encouraging. To keep your heart healthy, engage in regular physical activity, aiming for at least 150 minutes of moderate-intensity aerobic exercise per week. Adopt a heart-healthy diet, which includes plenty of fruits, vegetables, whole grains, lean proteins, and low-fat dairy products. Manage stress through relaxation techniques and maintain a healthy blood pressure level.")
            
        fig = st.plotly_chart(
                {
                    "data": [
                        {
                "values": [heart_probability, 1 - heart_probability],
                "labels": ["Disease", "No Disease"],
                "type": "pie",
                "hole": 0.5,
            }
        ],
            "layout": {"title": "Probability of Developing a Disease"},
    }
)
    
   
    
   
    
   st.success(heart_diagnosis)

    
    
    
    
if (selected == 'Diabetes Prediction'):
   st.title('Diabetes Prediction using ML')
   
   st.write("Welcome to our dedicated page on diabetes prediction using Machine Learning! Diabetes is a prevalent chronic condition affecting millions worldwide. However, with advancements in technology, particularly Machine Learning, we are now equipped with powerful tools to predict and prevent diabetes more accurately than ever before.")
   st.write("**Understanding Diabetes**") 
   st.write("**What is Diabetes?**")
   st.write("Diabetes is a metabolic disorder characterized by high blood sugar levels. Predicting its onset can significantly improve patient outcomes, making early intervention possible.")
   st.write("**How Machine Learning Helps**")
   st.write("Machine Learning algorithms analyze vast datasets, identifying patterns that might be invisible to the human eye. By processing factors such as lifestyle, genetics, and health history, these algorithms can predict the likelihood of an individual developing diabetes.")
   
   
   
   st.header("How to Use Our Diabetes Prediction Tool")
   st.write("To use our prediction tool, simply enter the required information in the form below, and our Machine Learning model will provide you with an assessment of your liver disease risk.")
   
   st.write("1. **Pregnancies:** Number of times pregnant.")
   st.write("2. **Glucose:** Plasma glucose concentration, a 2 hours in an oral glucose tolerance test.")
   st.write("3. **Blood Pressure:** Diastolic blood pressure (mm Hg).")
   st.write("4. **Skin Thickness:** Triceps skin fold thickness (mm).")
   st.write("5. **Insulin:** 2-Hour serum insulin (mu U/ml).")
   st.write("6. **BMI:** Body Mass Index.")
   st.write("7. **Diabetes pedigree function:** A function which scores likelihood of diabetes based on family history.")
   st.write("8. **Age(years)**")
   # st.write("To make the diagnosis you need to enter the following information: ")
   
   
   st.header("Enter the following parameters: ")
   
   
   # getting the input data from the user
   col1, col2, col3 = st.columns(3)
    
   with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
        
   with col2:
        Glucose = st.text_input('Glucose Level')
    
   with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    
   with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    
   with col2:
        Insulin = st.text_input('Insulin Level')
    
   with col3:
        BMI = st.text_input('BMI value')
    
   with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    
   with col2:
        Age = st.text_input('Age of the Person')
    
    
   # code for Prediction
   diab_diagnosis = ''
   input_data1 = (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
   input_data_as_numpy_array= np.asarray(input_data1)
   input_data_reshaped1 = input_data_as_numpy_array.reshape(1,-1)
   
    
    # creating a button for Prediction
    
   if st.button('Diabetes Test Result'):
        diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        diab_probability = diabetes_model.predict_proba(input_data_reshaped1)[0][1]
        if (diab_prediction[0] == 1):
          diab_diagnosis = 'Diabetes Risk Identified'
          st.write("It's crucial to start managing blood sugar levels promptly. Consult an endocrinologist or a diabetes specialist for a comprehensive diabetes management plan. This may include medications, insulin therapy, dietary adjustments, and regular physical activity. Monitoring blood sugar levels regularly and following the healthcare provider's recommendations can help manage diabetes effectively and prevent complications.")
        else:
          diab_diagnosis = 'Diabetes Risk Not Identified'
          st.write("A negative prediction for diabetes is a positive outcome. To reduce the risk of diabetes, maintain a healthy weight through regular exercise and a balanced diet. Focus on consuming whole foods, such as fruits, vegetables, whole grains, and lean proteins. Limit sugary foods and beverages, and avoid excessive consumption of processed carbohydrates. Regular physical activity helps in managing blood sugar levels and overall well-being.")
        
        fig = st.plotly_chart(
                {
                    "data": [
                        {
                "values": [diab_probability, 1 - diab_probability],
                "labels": ["Disease", "No Disease"],
                "type": "pie",
                "hole": 0.5,
            }
        ],
            "layout": {"title": "Probability of Developing a Disease"},
    }
)
        
        
        
        
   st.success(diab_diagnosis)
   
    
if (selected == 'Liver Disease Prediction'):
    st.title('Liver Disease Prediction using ML')
    
    st.write("Welcome to our Liver Disease Prediction page! Liver disease is a serious health condition that affects millions of people globally. Early prediction of liver disease is vital for timely medical intervention and improving the patient's quality of life. Our advanced Machine Learning model can help assess the risk of liver disease based on various factors. Let's explore how it works and how you can use it for your health assessment.")
    st.write("**Understanding Liver Disease**") 
    st.write("**What is Liver Disease?**")
    st.write("Liver disease refers to any condition that affects the liver and prevents it from functioning properly. It can be caused by various factors, including viruses, alcohol use, obesity, and certain medications. Early detection is essential for effective treatment and management.")
    st.write("**How Machine Learning Helps**")
    st.write("")
    
    
    st.header("How to Use Our Liver Disease Prediction Tool")
    st.write("To use our prediction tool, simply enter the required information in the form below, and our Machine Learning model will provide you with an assessment of your diabetes risk.")
    
    # st.write("To make the diagnosis you need to enter the following information: ")
    st.write("1. **Age:** Enter your age in years.")
    st.write("2. **Gender:** Select your gender (Male/Female). ")
    st.write("3. **Total Bilirubin:**  Enter your total bilirubin level in mg/dl.")
    st.write("4. **Direct Bilirubin:** Enter your direct bilirubin level in mg/dl.")
    st.write("5. **Alkaline Phosphatase:** Enter your alkaline phosphatase level in IU/L. ")
    st.write("6. **Alamine Aminotransferase:** Enter your alanine aminotransferase (ALT) level in IU/L.")
    st.write("7. **Aspartate Aminotransferase:** Enter your aspartate aminotransferase (AST) level in IU/L.")
    st.write("8. **Total Proteins:** Enter your total proteins level in g/dl.")
    st.write("9. **Albumin:** Enter your albumin level in g/dl.")
    st.write("10. **Albumin and Globulin Ratio:** Enter your albumin and globulin ratio.")
   
    
    st.header("Enter the following parameters: ")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
         Age = st.text_input('Enter you age')
         
    with col2:
         Gender = st.selectbox("Select Gender", ["Female", "Male"])
     
    with col3:
         Total_Bilirubin = st.text_input('Enter Total Bilirubin')
     
    with col1:
         Direct_Bilirubin = st.text_input('Enter Direct Bilirubin')
     
    with col2:
         Alkaline_Phosphotase = st.text_input('Alkaline Phosphotase')
         
    with col3:
         Alamine_Aminotransferase = st.text_input('Alamine Aminotransferase')
          
    with col1:
         Aspartate_Aminotransferase = st.text_input('Aspartate Aminotransferase')
     
    with col2:
         Total_Protiens = st.text_input('Total Protiens')
     
    with col3:
         Albumin = st.text_input('Albumin')
         
    with col1:
         Albumin_and_Globulin_Ratio = st.text_input('Albumin and Globulin Ratio')
         
         
    Gender = 1 if Gender == "Male" else 0
    
    liv_diagnosis = ''
    input_data2 = (Age, Gender, Total_Bilirubin, Direct_Bilirubin, Alkaline_Phosphotase,Alamine_Aminotransferase,Aspartate_Aminotransferase,  Total_Protiens, Albumin, Albumin_and_Globulin_Ratio )
    input_data_as_numpy_array= np.asarray(input_data2)
    input_data_reshaped2 = input_data_as_numpy_array.reshape(1,-1)
    
    
    if st.button('Liver Test Result'):
         liv_prediction = liver_model.predict([[Age,Gender,	Total_Bilirubin, Direct_Bilirubin, Alkaline_Phosphotase, Alamine_Aminotransferase, Aspartate_Aminotransferase, Total_Protiens, Albumin, Albumin_and_Globulin_Ratio]])
         liv_probability = liver_model.predict_proba(input_data_reshaped2)[0][1]
         if (liv_prediction[0] == 1):
           liv_diagnosis = 'Liver Disease Not Detected'
           st.write("If the prediction indicates no risk of liver disease, it's a positive sign for your liver health. To maintain a healthy liver, focus on adopting a balanced diet rich in fruits, vegetables, and whole grains. Limit alcohol consumption and avoid high-risk behaviors such as sharing needles, which can lead to liver infections. Regular exercise and maintaining a healthy weight also contribute to liver well-being.")
         else:
           liv_diagnosis = 'Liver Disease Detected'
           st.write("It's essential to consult a healthcare professional immediately. Early diagnosis is key to managing liver diseases effectively. Your doctor will conduct further tests to confirm the diagnosis and prescribe appropriate treatments. Additionally, adopting a healthy lifestyle, including a balanced diet and avoiding alcohol consumption, can significantly improve liver health.")
           
         fig = st.plotly_chart(
                {
                    "data": [
                        {
                "values": [liv_probability, 1 - liv_probability],
                "labels": ["Disease", "No Disease"],
                "type": "pie",
                "hole": 0.5,
            }
        ],
            "layout": {"title": "Probability of Developing a Disease"},
    }
)
            
         
            
    st.success(liv_diagnosis)
    
    
if (selected == 'Kidney Disease Prediction'):
    st.title('Kidney Disease Prediction using ML')
    
    st.write("Welcome to our Kidney Disease Prediction page! Chronic Kidney Disease (CKD) is a serious condition that affects millions of people worldwide. Early detection and intervention can significantly improve outcomes. Our advanced Machine Learning model analyzes various health indicators to predict the likelihood of kidney disease. Input your information, and let's assess your kidney health together.")
    st.write("**Understanding Kidney Disease**") 
    st.write("**What is Chronic Kidney Disease (CKD)?**")
    st.write("Chronic Kidney Disease is a gradual loss of kidney function over time. It can lead to complications and adversely affect your overall health.")
    st.write("**How Machine Learning Helps**")
    st.write("Machine Learning algorithms analyze vital health markers to identify patterns and assess the risk of kidney disease. By processing factors such as blood pressure, blood glucose, and urinary indicators, these algorithms provide valuable insights.")
    
    
    
    st.header("How to Use Our Kidney Disease Prediction Tool")
    st.write("To use our prediction tool, simply enter the required information in the form below, and our Machine Learning model will provide you with an assessment of your Kidney disease risk.")
    
    st.write("1. **Age:** Age of the individual in years.")
    st.write("2. **Blood Pressure:** Systolic blood pressure value in mm Hg.")
    st.write("3. **Specific Gravity:** Specific gravity of urine.")
    st.write("4. **Albumin:** Albumin levels in urine.")
    st.write("5. **Sugar:** Sugar levels in urine.")
    st.write("6. **Red Blood Cells:** Presence of red blood cells in urine.")
    st.write("7. **Pus Cell:** Presence of pus cells in urine.")
    st.write("8. **Pus Cell Clumps:** Clumping of pus cells in urine.")
    
    st.write("9. **Bacteria:** Presence of bacteria in urine.")
    st.write("10. **Blood Glucose Random:** Random blood glucose level.")
    st.write("11. **Blood Urea:** Blood urea levels.")
    st.write("12. **Serum Creatinine Level:** Serum creatinine level in mg/dL.")
    st.write("13. **Sodium:** Sodium levels in blood.")
    st.write("14. **Potassium:** Potassium levels in blood.")
    st.write("15. **Hemoglobin:** Hemoglobin levels in blood.")
    st.write("16. **Packed Cell Volume:** Volume occupied by packed red blood cells in blood.")
    
    st.write("17. **White Blood Cell Count:** Count of white blood cells in blood.")
    st.write("18. **Red Blood Cell Count:** Count of red blood cells in blood.")
    st.write("19. **Hypertension:** Presence of hypertension.")
    st.write("20. **Diabetes Mellitus:** Presence of diabetes mellitus.")
    st.write("21. **Coronary Artery Disease:** Presence of coronary artery disease.")
    st.write("22. **Appetite:** Patient's appetite (good, poor).")
    st.write("23. **Pedal Edema:** Presence of pedal edema")
    st.write("24. **Anemia:** Presence of anemia.")
    
    st.header("Enter the following parameters: ")
    
    
    col1, col2, col3 = st.columns(3)
     
    with col1:
         age = st.text_input('Enter your age')
         
    with col2:
         bp = st.text_input('Blood Pressure Value')
     
    with col3:
         sg = st.text_input('Specific Gravity')
     
    with col1:
         al = st.text_input('Albumin')
     
    with col2:
         su = st.text_input('Sugar Level')
     
    with col3:
         rbc = st.selectbox('Red Blood Cell',["Normal", "Abnormal"])
     
    with col1:
         pc = st.selectbox('Pus Cell',["Typical", "Atypical"])
    
         
    with col2: 
        pcc = st.selectbox('Pus Cell Clumps',["Present", "Not Present"])
     
    with col3:
          ba = st.selectbox('Bacteria in Urine',["Existing", "Not Existent"])
          
    with col1:
          bgr = st.text_input('Blood Glucose Random')
      
    with col2:
          bu = st.text_input('Blood Urea')
      
    with col3:
          sc = st.text_input('Serum Creatinine Level')
      
    with col1:
          sod = st.text_input('Sodium levels in blood')
      
    with col2:
          pot = st.text_input('Potassium levels in blood')
      
    with col3:
          hemo = st.text_input('Hemoglobin levels in blood')
      
    with col1:
          pcv = st.text_input('Packed Cell Volume')
          
    with col2: 
         wc = st.text_input('White Blood Cell Count')
         
    with col3:
         rc = st.text_input('Red Blood Cell Count')
       
    with col1:
         htn = st.selectbox('Presence of Hypertension', ["Yes", "No"])
           
    with col2: 
         dm = st.selectbox('Presence of diabetes mellitus', ["True", "False"])
   
    
   
    
    with col3:
          cad = st.selectbox('Coronary Artery Disease', ["Present", "Not Present"])
          
    with col1: 
         appet = st.selectbox('Appetite', ["Good", "Poor"])
         
    with col2:
         pe = st.selectbox('Presence of pedal edema', ["Yes", "No"])  
    with col3:
        ane = st.selectbox('Presence of anemia', ["Present", "Not Present"])
         
    
    rbc = 0 if rbc == "Normal" else 1
    pc = 0 if pc == "Typical" else 1
    pcc = 0 if pcc == "Present" else 1
    ba = 0 if pcc == "Existing" else 1
    htn = 0 if htn == "Yes" else 1
    dm = 0 if dm == "True" else 1
    cad = 0 if cad == "Present" else 1
    appet = 0 if appet == "Good" else 1
    pe = 0 if pe == "Yes" else 1
    ane = 0 if ane == "Yes" else 1
    
    
    
    kidney_diagnosis = ''
    input_data3 = (age,	bp,	sg,	al,	su,	rbc,	pc,	pcc	,ba,	bgr,	bu,	sc,	sod,	pot,	hemo,	pcv,	wc,	rc,	htn,	dm,	cad,	appet,	pe,	ane)

    input_data_as_numpy_array= np.asarray(input_data3)
    input_data_reshaped3 = input_data_as_numpy_array.reshape(1,-1)
    
    
    if st.button('Kidney Test Result'):
         kidney_prediction = kidney_model.predict([[age,	bp,	sg,	al,	su,	rbc,	pc,	pcc	,ba,	bgr,	bu,	sc,	sod,	pot,	hemo,	pcv,	wc,	rc,	htn,	dm,	cad,	appet,	pe,	ane]])
         kidney_probability = kidney_model.predict_proba(input_data_reshaped3)[0][1]
         if (kidney_prediction[0] == 1):
           kidney_diagnosis = 'Kidney Disease Detected'
           st.write("Our analysis indicates a potential risk of kidney disease based on the information provided. Kidney diseases can have various causes, such as high blood pressure, diabetes, or genetic factors. It's important to take this prediction seriously and seek professional medical advice for a comprehensive evaluation.")
         else:
           kidney_diagnosis = 'Kidney Disease Not Detected'
           st.write("Based on the information provided, our analysis does not indicate a risk of kidney disease at this time. It's important to maintain a healthy lifestyle to continue supporting your kidney health.")
           
         fig = st.plotly_chart(
                {
                    "data": [
                        {
                "values": [kidney_probability, 1 - kidney_probability],
                "labels": ["Disease", "No Disease"],
                "type": "pie",
                "hole": 0.5,
            }
        ],
            "layout": {"title": "Probability of Developing a Disease"},
    }
)
         st.success(kidney_diagnosis)