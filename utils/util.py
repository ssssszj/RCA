import os
import joblib
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.linear_model import LogisticRegression

logistic_regression_tool = {
    "type": "function",
    "function": {
        "name": "logistic_regression_analysis",
        "description": "Analyze feature coefficients using logistic regression.",
        "parameters": {
            "type": "object",
            "properties": {
                "data": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "features": {"type": "array", "items": {"type": "number"}},
                            "label": {"type": "integer"}
                        }
                    },
                    "description": "Input data with features and labels."
                },
                "feature_name":{
                    "type": "array",
                    "items": {"type":"string", "description":"name of features"}
                }
            },
            "required": ["data"]
        }
    }
}

decision_tree_tool = {
    "type": "function",
    "function": {
        "name": "decision_tree_analysis",
        "description": "Analyze feature importance using a decision tree.",
        "parameters": {
            "type": "object",
            "properties": {
                "data": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "features": {"type": "array", "items": {"type": "number"}},
                            "label": {"type": "integer"}
                        }
                    },
                    "description": "Input data with features and labels."
                },
                "feature_name":{
                    "type": "array",
                    "items": {"type":"string", "description":"name of features"}
                },
                "max_depth": {
                    "type": "integer",
                    "default": 3,
                    "description": "Maximum depth of the tree for interpretability."
                }
            },
            "required": ["data"]
        }
    }
}

def summarize_trial(agents):
    correct = [a for a in agents if a.is_correct()]
    incorrect = [a for a in agents if a.is_finished() and not a.is_correct()]
    return correct, incorrect

def confusion_matrix(correct,incorrect):
    tp = [a for a in correct if a.prediction == "catheter-related thrombosis"]
    tn = [a for a in correct if a.prediction == "no catheter-related thrombosis"]
    fp = [a for a in incorrect if a.prediction == "catheter-related thrombosis"]
    fn = [a for a in incorrect if a.prediction == "no catheter-related thrombosis"]
    return len(tp), len(tn), len(fp), len(fn)

def remove_fewshot(prompt: str) -> str:
    prefix = prompt.split('Here are some examples:')[0]
    suffix = prompt.split('(END OF EXAMPLES)')[1]
    return prefix.strip('\n').strip() + '\n\n' +  suffix.strip('\n').strip()

def remove_reflections(prompt: str) -> str:
    prefix = prompt.split('You have attempted to tackle the following task before and failed.')[0]
    suffix = prompt.split('\n\Features:')[-1]
    return prefix.strip('\n').strip() + '\n\Features' +  suffix.strip('\n').strip()

def log_trial(agents, trial_n):
    correct, incorrect = summarize_trial(agents)

    log = f"""
########################################
BEGIN TRIAL {trial_n}
Trial summary: Correct: {len(correct)}, Incorrect: {len(incorrect)}
#######################################
"""

    log += '------------- BEGIN CORRECT AGENTS -------------\n\n'
    for agent in correct:
        log += remove_fewshot(agent._build_agent_prompt()) + f'\nCorrect answer: {agent.target}\n\n'

    log += '------------- BEGIN INCORRECT AGENTS -----------\n\n'
    for agent in incorrect:
        log += remove_fewshot(agent._build_agent_prompt()) + f'\nCorrect answer: {agent.target}\n\n'

    return log

def save_agents(agents, dir: str):
    os.makedirs(dir, exist_ok=True)
    for i, agent in enumerate(agents):
        joblib.dump(agent, os.path.join(dir, f'{i}.joblib'))

def save_results(agents, dir: str):
    os.makedirs(dir, exist_ok=True)
    results = pd.DataFrame()
    for agent in agents:
        results = pd.concat([results, pd.DataFrame([{
                                        'Prompt': remove_fewshot(agent._build_agent_prompt()),
                                        'Response': agent.scratchpad.split('Disease Prediction:')[-1],
                                        'Target': agent.target
                                        }])], ignore_index=True)
    results.to_csv(dir + 'results.csv', index=False)

def logistic_regression_analysis(data, feature_name):
    X = [sample["features"] for sample in data]
    y = [sample["label"] for sample in data]

    model = LogisticRegression(max_iter=10000)
    try:
        model.fit(X,y)
    except Exception as e:
        print("Error in logistic regression fitting:", e)
        print("Data:", data)
        return {}

    importance = model.coef_[0]
    # print(importance)
    result = {}
    print(len(importance), len(feature_name))
    # print(feature_name)
    for i in range(len(importance)):
        result[feature_name[i]] = importance[i]

    return result

def decision_tree_analysis(data,feature_name,max_depth=3):
    X = [sample["features"] for sample in data]
    y = [sample["label"] for sample in data]

    model = DecisionTreeClassifier(max_depth=max_depth)
    try:
        model.fit(X,y)
    except Exception as e:
        print("Error in decision tree fitting:", e)
        print("Data:", data)
        return {}
    importance = model.feature_importances_
    # tree_rules = export_text(model, feature_names=[f'Feature {i}' for i in range(len(X[0]))])
    result = {}
    for i in range(len(importance)):
        result[feature_name[i]] = importance[i]
    return result

