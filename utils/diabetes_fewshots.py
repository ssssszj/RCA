PREDICT_EXAMPLES = """
Features: Number of pregnancies is 1, Plasma glucose concentration (2-hour test) level is 135, Diastolic blood pressure is 54 mm Hg, Triceps skin fold thickness is 0 mm, 2-Hour serum insulin level is 0 mu U/ml, BMI is 26.7, DiabetesPedigreeFunction(Genetic diabetes score) is 0.687, Age is 62.

Disease Prediction: no diabetes

Explanation: The patient's 2-hour plasma glucose level is 135 mg/dL, which is below the diagnostic threshold for diabetes (≥200 mg/dL) and even below the range for prediabetes (140-199 mg/dL). While factors like age (62), overweight BMI (26.7), and a moderate genetic risk score (0.687) increase diabetes risk, the absence of elevated glucose levels within diagnostic ranges and other features (e.g., low triceps skin fold thickness, low insulin level) do not meet criteria for diabetes. Diagnosis primarily relies on glucose levels, which here are within normal limits.

Features: Number of pregnancies is 4, Plasma glucose concentration (2-hour test) level is 171, Diastolic blood pressure is 72 mm Hg, Triceps skin fold thickness is 0 mm, 2-Hour serum insulin level is 0 mu U/ml, BMI is 43.6, DiabetesPedigreeFunction(Genetic diabetes score) is 0.479, Age is 26.

Disease Prediction: diabetes

Explanation: The patient's plasma glucose concentration (171 mg/dL) exceeds the prediabetes threshold (≥140 mg/dL) and approaches the diabetes range, combined with a markedly elevated BMI (43.6, class III obesity), a major risk factor for type 2 diabetes. The genetic risk score (0.479) and history of 4 pregnancies (potential gestational diabetes risk) further support this prediction. While triceps skinfold thickness and insulin levels of 0 suggest possible data anomalies, the high glucose and BMI strongly indicate diabetes likelihood despite the patient's younger age (26).
"""