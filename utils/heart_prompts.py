PREDICT_INSTRUCTION = """Given clinical features of patients, estimate whether the patient has the heart disease or not and explain your reasoning.  Give your response in this format:
(1) Disease Prediction, which should be either "no heart disease" or "heart disease".
(2) Explanation, which should be in a single, short paragraph.
Here are some examples:
{examples}
(END OF EXAMPLES)

Features:
{features}

Disease Prediction: """


# 对规则进行自反思。修改先前规则
MODIFY_GUIDELINE_INSTRUCTION = """You are an advanced reasoning agent that can improve based on self reflection. The original task is "Given clinical features of patient, estimate whether the patient has the heart disease or not and explain your reasoning.". Now you will be given the previous rules and some wrong samples that you have attempted to predict heart disease but failed. Considering patients' clinical features and their true heart disease results, you need to reflect on and revise rules to help heart disease prediction.\\
The rules must be supported by medical knowledge. Note that rules like "If the patient has a blood pressure greater than 160 and has diabetes, predict no heart disease." are not reasonable -- you cannot predict heart disease only based on blood pressure and diabetes.\\
There might be some outliers in the data. You will also be given the features distribution on the whole dataset. You can determine the outliers based on distribution before summarizing the rules. Don't utilize relationship between outliers and heart disease, and don't let your rules be destroyed by outliers easily. Don't exclude patients with outliers, you should use other features to predict heart disease for them.\\
Keep the rules brief. Only output rules. Your rules must be general enough for any patients. Give your response in this format:
Rules, which should be a list of rules, each rule is a short sentence.

Data distribution:
{distribution}

Previous rules:
{rules}
(If it is empty, it means summarizing the initial rules)

Wrong samples:
{samples}

Rules:
"""

# 每个epoch总结一次
SUMMARIZE_REVISE_INSTRUCTION = """You are an advanced reasoning agent that can improve based on self reflection. The original task is "Given clinical features of tumor patient and some prediction rules, estimate whether the patient has the heart disease or not and explain your reasoning" Given the previous rules and the features distribution, you need to check and delete the error rules.
Rules like "If the patient has a blood pressure greater than 160 and has diabetes, predict no heart disease."  are not reasonable, because it's inconsistent with medical knowledge -- you cannot predict heart disease only based on blood pressure and diabetes.
Also, there might be some outliers in data. Rules that utilize relationship between outliers and disease, like "If the patient has a CRP level between 10 and 12, but any numerical feature is an extreme outlier, they are less likely to develop heart disease." is forbidden. However, outliers could mislead prediction, so you should indicate in the rules how to identify outliers. Don't exclude patients with outliers, you should use other features to support disease prediction for them.
And rules that are too specific for certain patient are awful. You need to delete rules similar to those listed above. Give your response in this format:
Rules, which should be a list of rules, each rule is a short sentence.

Previous distribution:
{distribution}

Previous rules:
{rules}

Rules:
"""

# 基于规则进行预测
PREDICT_GUIDE_INSTRUCTION = """Given clinical features of tumor patient, estimate whether the patient has the heart disease or not and explain your reasoning. You will be given some rules for prediction and distribution of training dataset. You can refer to the following rules, but don't limit yourself to them. Remember there are some outliers in the data. Give your response in this format:
(1) Disease Prediction, which should be either "no heart disease" or "heart disease".
(2) Explanation, which should be in a single, short paragraph.

Here are some examples:
{examples}
(END OF EXAMPLES)

Here are some rules:
{rules}
(If it is empty, it means there is no rule.)
(END OF RULES)

Here is the distribution:
{distribution}
(END OF DISTRIBUTION)

Features:
{features}

Disease Prediction: 
"""

# 让LLM选择直接给出规则还是调用函数
AGENT_ANALYSE_INSTRUCTION = """Given clinical features of a group of patients and their heart disease labels, analyze each feature in relation to the heart disease, and gives an ordering of the features in decreasing order of importance. You will also be provided with a ordering based on the previous samples. Please write code to implement my task, but do not need to show any of your code in the output. I implemented a logistic regression function and a decision tree function that will provide an the statistical importance of the features, just tell me if you think it's necessary to call it and get the results. But note that if there is only one class of samples, it is not possible to call the functions, so carefully check the labels. There are two scenarios for your answer:
(1) If you think it's necessary to call the function, output "logistic regression needed" or "decision tree needed". Extract the feature values, labels and feature names in the text and organize them into a json object. The object has two attributes: "data" and "feature_names". “data” is a json array, in which each object has a “features” attribute, a “label” attribute, respectively, to store the feature values and labels of each sample. The “features” attribute is a list of values only. You need to express the category characteristics as numbers, e.g. 0 for male and 1 for female. And the "label" is a binary value for disease status, 0 for no disease and 1 for disease. "feature_names" is a list containing the feature names. There should be 19 features. If there is any feature unknown in the sample, you need to decide for yourself what values to fill in, don't just put "None". Don't output anything redundant.
(2) If you think it's not necessary to call the function, just output the new ordering. Start with "Feature Importance Ordering" and then list the features in decreasing order of importance. Rememeber not to change feature names and add or remove features, just reorder them. If there is only one class of samples, you must choose this scenario.
Here is previous ordering:
{order}
(END OF ORDERING)

Features and labels:
{features}
"""

# 直接分析排序
ORDER_ANALYSE_INSTRUCTION = """Given clinical features of a group of patients and their heart disease labels, analyze each feature in relation to the heart disease, and gives an ordering of the features in decreasing order of importance. There are some abnormal values, you need to judge for yourself. You will also be provided with a ordering based on the previous samples. Please write code to implement my task, but do not need to show any of your code in the output. 
Just output the new ordering. Your answer should start with "Feature Importance Ordering" and then list the features in decreasing order of importance. Rememeber not to change feature names and add or remove features, just reorder them. 
Here is previous ordering:
{order}
(END OF ORDERING)

Features and labels:
{features}
"""

# 解析调用函数并返回结果
FUNCTION_ANALYSE_INSTRUCTION = """Given clinical features of a group of patients and their heart disease labels, analyze each feature in relation to the heart disease, and gives an ordering of the features in decreasing order of importance. There are some abnormal values, you need to judge for yourself. You will also be provided with a ordering based on the previous samples. Please write code to implement my task, but do not need to show any of your code in the output. You will also be provided with logistic regression coefficients or decision tree feature importance, which is calculated using data in texts.
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
AGENT_PREDICT_INSTRUCTION = """Given clinical features of tumor patient, estimate whether the patient has the heart disease or not and explain your reasoning. You will be provided a feature importance ordering, which you can refer to. Give your response in this format:
(1) Disease Prediction, which should be either "no heart disease" or "heart disease".
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

O3_PREDICT_INSTRUCTION = """Given clinical features of a group of patients, estimate whether the patient has the heart disease or not and explain your reasoning seperately. For each patient, Give your response in this format:
ID: xxx, which should be the sample id.
Disease Prediction: xxx, which should be either "no heart disease" or "heart disease". No other options, no "low/medium/high heart disease".
Explanation: xxx, which should be in a single, short paragraph.
Here are some examples:
{examples}
(END OF EXAMPLES)

Input:
{features}

"""

O3_ORDER_PREDICT_INSTRUCTION = """Given clinical features of a group of patients, estimate whether the patient has the heart disease or not and explain your reasoning seperately.  You will be provided a feature importance ordering, which you can refer to. For each patient, Give your response in this format:
ID: xxx, which should be the sample id.
Disease Prediction: xxx, which should be either "no heart disease" or "heart disease". No other options, no "low/medium/high heart disease".
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
