import openai
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt, # type: ignore
    wait_random_exponential, # type: ignore
)
# from fastchat.model import get_conversation_template
import torch
import os
from utils.util import logistic_regression_analysis, decision_tree_analysis, logistic_regression_tool, decision_tree_tool
import json

class OpenAILLM:
    def __init__(self):
        self.model = "gpt-4o-mini"

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def __call__(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        completion = openai.chat.completions.create(model=self.model, messages=messages)
        response = completion.choices[0].message.content
        return response

class PipeLLM:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),  
            base_url="http://100.70.129.58:1025/v1",  #qwen
            # base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  #aliyun
        )
        self.model = "Qwen2.5-72B-Instruct"
        # self.model = "deepseek-r1"
    
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def __call__(self,prompt):
        messages = [{"role": "user", "content": prompt}]
        completion = self.client.chat.completions.create(model=self.model, messages=messages)
        response = completion.choices[0].message.content
        return response
    
class AgentLLM:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),  
            base_url="http://100.70.129.58:1025/v1",  #qwen
            # base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  #aliyun
        )
        self.model = "Qwen2.5-72B-Instruct"
        # self.model = "deepseek-r1"
    
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def __call__(self,prompt):
        messages = [{"role": "user", "content": prompt}]
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            )
        
        response = completion.choices[0].message.content
        return response
