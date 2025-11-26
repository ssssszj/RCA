PREDICT_INSTRUCTION = """Given clinical features of tumor patient, estimate whether the patient has the catheter related thrombosis(CRT) or not and explain your reasoning. Give your response in this format:
(1) Disease Prediction, which should be either "no catheter-related thrombosis" or "catheter-related thrombosis".
(2) Explanation, which should be in a single, short paragraph.
Here are some examples:
{examples}
(END OF EXAMPLES)

Features:
{features}

Disease Prediction: """

# 对规则进行自反思。修改先前规则
MODIFY_GUIDELINE_INSTRUCTION = """You are an advanced reasoning agent that can improve based on self reflection. The original task is " Given clinical features of tumor patient, estimate whether the patient has the catheter related thrombosis(CRT) or not and explain your reasoning. ". Now you will be given the previous rules and some wrong samples that you have attempted to predict CRT but failed. Considering patients' clinical features and their true CRT results, you need to reflect on and revise rules to help CRT prediction.
The rules must be supported by medical knowledge. Note that rules like "If the patient has a BMI value between 10 and 50 and a history of previous CRT, predict no catheter-related thrombosis." are not reasonable -- you cannot predict CRT only based on BMI and previous CRT history.
There might be some outliers in the data. You will also be given the features distribution on the whole dataset. You can determine the outliers based on distribution before summarizing the rules. Don't utilize relationship between outliers and CRT, and don't let your rules be destroyed by outliers easily. Don't exclude patients with outliers, you should use other features to predict CRT for them.
Keep the rules brief. Only output rules. Your rules must be general enough for any patients. Give your response in this format:
Rules, which should be a list of rules, each rule is a short sentence. 

Here are input.
Data distribution:
{distribution}

Previous rules:
{rules}
(If it is empty, it means summarizing the initial rule)

Wrong samples:
{samples}

Rules:
"""

# 每个epoch总结一次
SUMMARIZE_REVISE_INSTRUCTION = """You are an advanced reasoning agent that can improve based on self reflection. The original task is "Given clinical features of tumor patient and some prediction rules, estimate whether the patient has the catheter related thrombosis(CRT) or not and explain your reasoning" Given the previous rules and the features distribution, you need to check and delete the error rules.
Rules like "If the patient has a BMI value between 10 and 50 and a history of previous CRT, predict no catheter-related thrombosis." are not reasonable, because it's inconsistent with medical knowledge -- you cannot predict CRT only based on BMI and previous CRT history.
Also, there might be outliers in data, which is introduced randomly. Rules that utilize relationship between outliers and disease, like"If the patient has a D-dimer level between 0.1 and 0.79, but any numerical feature is an extreme outlier, they are less likely to develop CRT." is forbidden. However, outliers could mislead prediction, so you should indicate in the rules how to identify outliers. Don't exclude patients with outliers, you should use other features to predict CRT for them.
And rules that are too specific for certain patient are awful. You must delete rules similar to those listed above. Give your response in this format:
Rules, which should be a list of rules, each rule is a short sentence.

Here are input.
Data distribution:
{distribution}

Previous rules:
{rules}

Rules:
"""

# 基于规则进行预测
PREDICT_GUIDE_INSTRUCTION = """Given clinical features of tumor patient, estimate whether the patient has the catheter related thrombosis(CRT) or not and explain your reasoning. You will be given some rules for prediction and distribution of training dataset. You can refer to the following rules, but don't limit yourself to them. Give your response in this format:
(1) Disease Prediction, which should be either "no catheter-related thrombosis" or "catheter-related thrombosis". No other options, no "low/medium/high catheter-related thrombosis", no "CRT" or any abbreviations.
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
