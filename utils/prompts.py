PREDICT_INSTRUCTION = """Given clinical features of tumor patient, estimate whether the patient has the catheter related thrombosis(CRT) or not  nd explain your reasoning. Give your response in this format:
(1) CRT prediction, which should be either "no catheter-related thrombosis" or "catheter-related thrombosis".
(2) Explanation, which should be in a single, short paragraph.
Here are some examples:
{examples}
(END OF EXAMPLES)

Features:
{features}

CRT Prediction: """

# 对规则进行自反思。生成初始规则
ORIGIN_GUIDELINE_INSTRUCTION = """You are an advanced analytics agent that can generate rules for prediction based on a few samples. The original task is "Given clinical features of tumor patient, estimate whether the patient has the catheter related thrombosis(CRT) or not.". You will be given some sample patients. Considering patients' clinical features and their CRT results, generate brief rules that can guide CRT risk prediction. Only output the rules. Your rules must be general enough for any patients. Limit your answer within 256 tokens.

samples:
{samples}

Rules:
"""

# 对规则进行自反思。修改先前规则
MODIFY_GUIDELINE_INSTRUCTION = """You are an advanced reasoning agent that can improve based on self refection. The original task is "Given clinical features of tumor patient and some prediction rules, estimate whether the patient has the catheter related thrombosis(CRT) or not.". Now you will be given the previous rules and some wrong samples that have failed previous CRT predictions. Considering patients' clinical features and their true CRT results, you need to reflect on and summarize the mistakes in the rules, and revise the rules to help CRT prediction. Try to analyze the effect that different inter-feature combinations have on the CRT. Keep the rules brief. Only output rules. Your rules must be general enough for any patients. Limit your answer within 256 tokens.
Previous rules:
{guidelines}

Wrong samples:
{samples}

Rules:
"""

# 基于规则进行预测
PREDICT_GUIDE_INSTRUCTION = """Given clinical features of tumor patient, estimate whether the patient has the catheter related thrombosis(CRT) or not and explain your reasoning. And gives an ordering of the features in decreasing order of importance. You will be given some rules for prediction. You can refer to the following rules, but don't limit yourself to them. Give your response in this format:
(1) CRT Prediction, which should be either "no catheter-related thrombosis" or "catheter-related thrombosis".
(2) Explanation, which should be in a single, short paragraph.
(3) Feature Importance Ordering, which should be a list of features in decreasing order of importance. Rememeber not to change feature names and add or remove features, just reorder them. There should be 16 features.
Here are some examples:
{examples}
(END OF EXAMPLES)
Here are some rules:
{rules}
(END OF RULES)

Features:
{features}

CRT Prediction: 
"""

# 让LLM选择直接给出规则还是调用函数
AGENT_ANALYSE_INSTRUCTION = """Given clinical features of a group of tumor patients and their catheter related thrombosis(CRT) labels, analyze each feature in relation to the CRT, and gives an ordering of the features in decreasing order of importance. You will also be provided with a ordering based on the previous samples. Please write code to implement my task, but do not need to show any of your code in the output. I implemented a logistic regression function and a decision tree function that will provide an the statistical importance of the features, just tell me if you think it's necessary to call it and get the results. But note that if there is only one class of samples, it is not possible to call the functions, so carefully check the labels. There are two scenarios for your answer:
(1) If you think it's necessary to call the function, output "logistic regression needed" or "decision tree needed". Extract the feature values and labels in the text and organize them into a json array. The name of the array is “data”, in which each object has a “feature” attribute, a “label” attribute, respectively, to store the feature values and labels of each sample. The “feature” attribute is a list of values only. You need to express the category characteristics as numbers, e.g. 0 for male and 1 for female. The feature names are extracted sequentially and output as a list named "feature_names." There should be 16 features. If there is any feature missing in the sample, you need to decide for yourself what values to fill in.
(2) If you think it's not necessary to call the function, just output the new ordering. Start with "Feature Importance Ordering" and then list the features in decreasing order of importance. Rememeber not to change feature names and add or remove features, just reorder them. If there is only one class of samples, you must choose this scenario.
Here is previous ordering:
{order}
(END OF ORDERING)

Features and labels:
{features}
"""

ORDER_ANALYSE_INSTRUCTION = """Given clinical features of a group of tumor patients and their catheter related thrombosis(CRT) labels, analyze each feature in relation to the CRT, and gives an ordering of the features in decreasing order of importance. You will also be provided with a ordering based on the previous samples. Please write code to implement my task, but do not need to show any of your code in the output. 
Just output the new ordering. Your answer should start with "Feature Importance Ordering" and then list the features in decreasing order of importance. Rememeber not to change feature names and add or remove features, just reorder them. 
Here is previous ordering:
{order}
(END OF ORDERING)

Features and labels:
{features}
"""

# 解析调用函数并返回结果
FUNCTION_ANALYSE_INSTRUCTION = """Given clinical features of a group of tumor patients and their catheter related thrombosis(CRT) labels, analyze each feature in relation to the CRT, and gives an ordering of the features in decreasing order of importance. You will also be provided with a ordering based on the previous samples. Please write code to implement my task, but do not need to show any of your code in the output. You will also be provided with logistic regression coefficients or decision tree feature importance, which is calculated using data in texts.
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
AGENT_PREDICT_INSTRUCTION = """Given clinical features of tumor patient, estimate whether the patient has the catheter related thrombosis(CRT) or not and explain your reasoning. You will be provided a feature importance ordering, which you can refer to. Give your response in this format:
(1) CRT Prediction, which should be either "no catheter-related thrombosis" or "catheter-related thrombosis".
(2) Explanation, which should be in a single, short paragraph.
Here are some examples:
{examples}
(END OF EXAMPLES)

Here is the ordering:
{order}
(END OF ORDERING)

Features:
{features}

CRT Prediction: 
"""

