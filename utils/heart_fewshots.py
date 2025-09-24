PREDICT_EXAMPLES = """
Features: Age is 62.0, Gender is Female, Blood Pressure is 133.0, Cholesterol Level is 166.0, Exercise Habits is Medium, Smoking is No, Family Heart Disease is No, Diabetes is No, BMI is 25.739170533963147, High Blood Pressure is No, Low HDL Cholesterol is Yes, High LDL Cholesterol is No, Stress Level is Low, Sleep Hours is 5.493276805328829, Sugar Consumption is Medium, Triglyceride Level is 126.0, Fasting Blood Sugar is 102.0, CRP Level is 11.60991435489297, Homocysteine Level is 8.297757016065253

Disease Prediction: heart disease

Explanation: The patient has several risk factors for heart disease. At 62 years old, the patient has a cholesterol level of 166, and despite having normal blood pressure according to the "High Blood Pressure" marker, a blood pressure of 133 is relatively close to the elevated range. The presence of low HDL cholesterol is a risk factor for heart disease. The C-reactive protein (CRP) level of 11.60991435489297 is elevated, indicating possible inflammation in the body, which is associated with heart disease. Although the patient has a medium level of exercise and no family history of heart disease or diabetes, the combination of age, low HDL cholesterol, and elevated CRP level increases the likelihood of having heart disease.


Features: Age is 35.0, Gender is Male, Blood Pressure is 159.0, Cholesterol Level is 261.0, Exercise Habits is Low, Smoking is No, Family Heart Disease is No, Diabetes is Yes, BMI is 21.63849835899007, High Blood Pressure is No, Low HDL Cholesterol is Yes, High LDL Cholesterol is No, Stress Level is High, Sleep Hours is 4.296875738592791, Sugar Consumption is Medium, Triglyceride Level is 385.0, Fasting Blood Sugar is 136.0, CRP Level is 1.9462702594315329, Homocysteine Level is 11.140952179886469

Disease Prediction: no heart disease

Explanation:  Although the patient presented with multiple risk factors such as elevated blood pressure, high cholesterol levels, diabetes, high triglycerides, high stress, low sleep hours, elevated CRP, and low HDL cholesterol, it has been determined that he has no heart disease. It is possible that there are mitigating factors not mentioned, such as effective medical management or significant lifestyle changes that reduce the impact of these risk factors on the heart.

"""