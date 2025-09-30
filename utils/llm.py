import openai
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt, # type: ignore
    wait_random_exponential, # type: ignore
    retry_if_exception_type,
)
# from fastchat.model import get_conversation_template
import torch
import os
from utils.util import logistic_regression_analysis, decision_tree_analysis, logistic_regression_tool, decision_tree_tool
import json
import time
import httpx

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
        self.model = "qwen2.5-72b-instruct"
        # self.model = "deepseek-r1"
        # self.model = "deepseek-v3-64k-local-preview"
        # self.model = "o3-mini"
        # self.model = "gpt-4.1-2025-04-14"
        # self.model = "gpt-4o-2024-08-06"

        self.request_interval = 5
        self.last_request_time = 0
        
        transport = httpx.HTTPTransport(
            limits=httpx.Limits(max_connections=100), 
            retries=3
        )

        timeout = httpx.Timeout(
            connect=10,  
            read=60,     
            write=10,    
            pool=5       
        )

        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),  
            # api_key=os.getenv("AIML_API_KEY"),
            # api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/",  # qwen
            # base_url="https://api.openai.com/v1/",  # openai
            # base_url="https://api.aimlapi.com/v1", # aiml
            # http_client=httpx.Client(transport=transport, timeout=timeout)
        )
    
    @retry(
        wait=wait_random_exponential(min=1, max=120), 
        stop=stop_after_attempt(10),  
        retry=(
            retry_if_exception_type(openai.RateLimitError) | 
            retry_if_exception_type(openai.APITimeoutError) |
            retry_if_exception_type(httpx.ReadTimeout) |
            retry_if_exception_type(openai.APIError)  
        )
    )
    def __call__(self, prompt):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.request_interval:
            time.sleep(self.request_interval - elapsed)
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                timeout=60,  
            )
            response = completion.choices[0].message.content
        finally:
            self.last_request_time = time.time()  
            
        return response

    
class AgentLLM:
    def __init__(self):
        # self.model = "qwen2.5-72b-instruct"
        # self.model = "deepseek-r1"
        # self.model = "deepseek-v3-64k-local-preview"
        # self.model = "gpt-4.1-2025-04-14"
        # self.model = "o3-mini"
        self.model = "qwen/qwen3-30b-a3b"
        # self.model = "qwen/qwen3-235b-a22b-2507"
        # self.model="deepseek/deepseek-chat-v3.1:free"
        # self.model="openai/o4-mini"
        # self.model="o4-mini-2025-04-16"
        # self.model="gpt-5-2025-08-07"
        # self.model = "gpt-4o-2024-08-06"


        self.request_interval = 15
        self.last_request_time = 0
        
        transport = httpx.HTTPTransport(
            limits=httpx.Limits(max_connections=100), 
            retries=3
        )

        timeout = httpx.Timeout(
            connect=10,  
            read=60,     
            write=10,    
            pool=5       
        )

        self.client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),  
            # api_key=os.getenv("DASHSCOPE_API_KEY"),
            # api_key=os.getenv("AIML_API_KEY"),
            # api_key=os.getenv("OPENAI_API_KEY"),
            # base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/",  # qwen
            # base_url="https://api.openai.com/v1/",  # openai
            # base_url="https://api.aimlapi.com/v1", # aiml
            base_url="https://openrouter.ai/api/v1", 
            # http_client=httpx.Client(transport=transport, timeout=timeout)
        )
    
    @retry(
        wait=wait_random_exponential(min=1, max=120), 
        stop=stop_after_attempt(3),  
        retry=(
            retry_if_exception_type(openai.RateLimitError) | 
            retry_if_exception_type(openai.APITimeoutError) |
            retry_if_exception_type(httpx.ReadTimeout) |
            retry_if_exception_type(openai.PermissionDeniedError) |
            retry_if_exception_type(openai.APIError)  
        )
    )
    def __call__(self, prompt):
        max_retry = 5
        elapsed = time.time() - self.last_request_time
        if elapsed < self.request_interval:
            time.sleep(self.request_interval - elapsed)
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                timeout=120,  
            )
            
            response = completion.choices[0].message.content
            tried_time = 1
            while response.strip() == "" and tried_time < max_retry:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    timeout=120,  
                )
                response = completion.choices[0].message.content
                tried_time += 1
            
        finally:
            self.last_request_time = time.time()  
        return response


