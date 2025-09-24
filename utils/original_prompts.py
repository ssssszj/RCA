PREDICT_INSTRUCTION = """Given clinical features of tumor patient, estimate whether the patient has the catheter related thrombosis(CRT) or not and explain your reasoning. Give your response in this format:
(1) Disease Prediction, which should be either "no catheter-related thrombosis" or "catheter-related thrombosis". No other options, no "low/medium/high catheter-related thrombosis".
(2) Explanation, which should be in a single, short paragraph.
Here are some examples:
{examples}
(END OF EXAMPLES)

Features:
{features}

Disease Prediction: """

# 对规则进行自反思。生成初始规则
ORIGIN_GUIDELINE_INSTRUCTION = """You are an advanced analytics agent that can generate rules for prediction based on a few samples. The original task is "Given clinical features of tumor patient, estimate whether the patient has the catheter related thrombosis(CRT) or not.". You will be given some sample patients. Considering patients' clinical features and their CRT results, generate brief rules that can guide CRT risk prediction. Remember that outliers don't give you any information because they are introduced randomly. Only output the rules. Your rules must be general enough for any patients.

samples:
{samples}

Rules:
"""

# 对规则进行自反思。修改先前规则
MODIFY_GUIDELINE_INSTRUCTION = """You are an advanced reasoning agent that can improve based on self refection. The original task is "Given clinical features of tumor patient and some prediction rules, estimate whether the patient has the catheter related thrombosis(CRT) or not.". Now you will be given the previous rules and some wrong samples that you have attempted to predict CRT but failed. Considering patients' clinical features and their true CRT results, you need to reflect on and summarize the mistakes in the rules, and revise them to help CRT prediction. Remember that outliers don't give you any information because they are introduced randomly. Keep the rules brief, and you can delete rules that you think is unnecessary. Only output rules. Your rules must be general enough for any patients.
Previous rules:
{guidelines}

Wrong samples:
{samples}

Rules:
"""

# 基于规则进行预测
PREDICT_GUIDE_INSTRUCTION = """Given clinical features of tumor patient, estimate whether the patient has the catheter related thrombosis(CRT) or not and explain your reasoning. You will be given some rules for prediction. You can refer to the following rules, but don't limit yourself to them. Give your response in this format:
(1) Disease Prediction, which should be either "no catheter-related thrombosis" or "catheter-related thrombosis". No other options, no "low/medium/high catheter-related thrombosis".
(2) Explanation, which should be in a single, short paragraph.
Here are some examples:
{examples}
(END OF EXAMPLES)
Here are some rules:
{rules}
(END OF RULES)

Features:
{features}

Disease Prediction: 
"""

# 让LLM选择直接给出规则还是调用函数
AGENT_ANALYSE_INSTRUCTION = """Given clinical features of a group of tumor patients and their catheter related thrombosis(CRT) labels, analyze each feature in relation to the CRT, and gives an ordering of the features in decreasing order of importance.   You will also be provided with a ordering based on the previous samples. Please write code to implement my task, but do not need to show any of your code in the output. I implemented a logistic regression function and a decision tree function that will provide an the statistical importance of the features, just tell me if you think it's necessary to call it and get the results. But note that if there is only one class of samples, it is not possible to call the functions, so carefully check the labels. There are two scenarios for your answer:
(1) If you think it's necessary to call the function, output "logistic regression needed" or "decision tree needed". Extract the feature values, labels and feature names in the text and organize them into a json object. The object has two attributes: "data" and "feature_names". “data” is a json array, in which each object has a “features” attribute, a “label” attribute, respectively, to store the feature values and labels of each sample. The “features” attribute is a list of values only. And the "label" is a binary value for disease status, 0 for no disease and 1 for disease. You need to express the category characteristics as numbers, e.g. 0 for male and 1 for female. "feature_names" is a list containing the feature names. There are 16 features. If there is any feature unknown in the sample, you need to decide for yourself what values to fill in, don't just put "None". Don't output anything redundant.
(2) If you think it's not necessary to call the function, just output the new ordering. Start with "Feature Importance Ordering" and then list the features in decreasing order of importance. Rememeber not to change feature names and add or remove features, just reorder them. If there is only one class of samples, you must choose this scenario.
Don't output anything redundant.
Here is previous ordering:
{order}
(END OF ORDERING)

Features and labels:
{features}
"""

# 直接分析排序
ORDER_ANALYSE_INSTRUCTION = """Given clinical features of a group of tumor patients and their catheter related thrombosis(CRT) labels, analyze each feature in relation to the CRT, and gives an ordering of the features in decreasing order of importance.   You will also be provided with a ordering based on the previous samples. Please write code to implement my task, but do not need to show any of your code in the output.  
Just output the new ordering. Your answer should start with "Feature Importance Ordering" and then list the features in decreasing order of importance. Rememeber not to change feature names and add or remove features, just reorder them. 
Here is previous ordering:
{order}
(END OF ORDERING)

Features and labels:
{features}
"""

# 解析调用函数并返回结果
FUNCTION_ANALYSE_INSTRUCTION = """Given clinical features of a group of tumor patients and their catheter related thrombosis(CRT) labels, analyze each feature in relation to the CRT, and gives an ordering of the features in decreasing order of importance.   You will also be provided with a ordering based on the previous samples. Please write code to implement my task, but do not need to show any of your code in the output. You will also be provided with logistic regression coefficients or decision tree feature importance, which is calculated using data in texts.  
Your answer should be in this format: Start with "Feature Importance Ordering" and then list the features in decreasing order of importance. Rememeber not to change feature names and add or remove features, just reorder them.

Here is previous ordering:
{order}
(END OF ORDERING)

Features and labels:
{features}

{function_type}
{function_result}

"""

# LLM利用排序直接预测
AGENT_PREDICT_INSTRUCTION = """Given clinical features of tumor patient, estimate whether the patient has the catheter related thrombosis(CRT) or not and explain your reasoning.   You will be provided a feature importance ordering, which you can refer to.   Give your response in this format:
(1) Disease Prediction, which should be either "no catheter-related thrombosis" or "catheter-related thrombosis".
(2) Explanation, which should be in a single, short paragraph.
Here are some examples:
{examples}
(END OF EXAMPLES)

Here is the ordering:
{order}
(END OF ORDERING)

Features:
{features}

Disease Prediction: 
"""

O3_PREDICT_INSTRUCTION = """Given clinical features of a group of tumor patients, estimate whether the patient has the catheter related thrombosis(CRT) or not and explain your reasoning seperately. For each patient, Give your response in this format:
ID: xxx, which should be the sample id.
Disease Prediction: xxx, which should be either "no catheter-related thrombosis" or "catheter-related thrombosis". No other options, no "low/medium/high catheter-related thrombosis".
Explanation: xxx, which should be in a single, short paragraph.
Here are some examples:
{examples}
(END OF EXAMPLES)

Input:
{features}

"""

O3_ORDER_PREDICT_INSTRUCTION = """Given clinical features of a group of tumor patients, estimate whether the patient has the catheter related thrombosis(CRT) or not and explain your reasoning seperately. You will be provided a feature importance ordering, which you can refer to. For each patient, Give your response in this format:
ID: xxx, which should be the sample id.
Disease Prediction: xxx, which should be either "no catheter-related thrombosis" or "catheter-related thrombosis". No other options, no "low/medium/high catheter-related thrombosis".
Explanation: xxx, which should be in a single, short paragraph.
Here are some examples:
{examples}
(END OF EXAMPLES)

Here is the ordering:
{order}
(END OF ORDERING)

Input:
{features}

"""

O3_CODE_PREDICT_INSTRUCTION = """
Given clinical features of a group of tumor patients, estimate whether the patient has the catheter related thrombosis(CRT) or not and explain your reasoning seperately. The file contains a group of tumor patients' clinical features and their catheter related thrombosis(CRT) labels. Analyze each feature in relation to the CRT, and predict testing samples based on your analysis.  I want you to write code to implement the task, but do not show any code in your response. For each patient, Give your response in this format:
ID: xxx, which should be the sample id.
Disease Prediction: xxx, which should be either "no catheter-related thrombosis" or "catheter-related thrombosis". No other options, no "low/medium/high catheter-related thrombosis".
Explanation: xxx, which should be in a single, short paragraph.

Features:
"""

O3_EXPLANATION_GENERATION = """
You are an advanced reasoning agent that can generate explanations for prediction task. The original task is "Given clinical features of tumor patient, estimate whether the patient has the catheter related thrombosis(CRT) or not". Now I'll give you the patient features, the predictions from the Lasso regression, and the descending order of the features by Lasso's feature coefficients. I want you to generate an explanation for the predictions based on this information. I will also provide some examples, follow the examples to generate your explanation. Give your response in this format:
(1)ID: xxx, which be the sample id.
(2)Explanation: xxx, which should be in a short paragraph.

Order:
{order}
(END OF ORDER)

Examples:
{examples}
(END OF EXAMPLES)

Input:
{input}
"""