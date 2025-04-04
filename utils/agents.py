from typing import List, Union, Literal
from utils.llm import OpenAILLM,PipeLLM, AgentLLM
from utils.prompts import PREDICT_GUIDE_INSTRUCTION, PREDICT_INSTRUCTION, AGENT_PREDICT_INSTRUCTION
from utils.fewshots import PREDICT_EXAMPLES


class PredictAgent:
    def __init__(self,
                 ticker: str,
                 features: str,
                 target: str,
                 predict_llm = PipeLLM()
                 ) -> None:

        self.ticker = ticker
        self.features = features
        self.target = target
        self.prediction = ''
        self.rule_prompt = PREDICT_GUIDE_INSTRUCTION
        self.predict_prompt = PREDICT_INSTRUCTION
        self.predict_examples = PREDICT_EXAMPLES
        self.agent_prompt = AGENT_PREDICT_INSTRUCTION
        self.predict_llm = predict_llm
        self.agent_llm = AgentLLM()
        self.llm = AgentLLM()

        self.__reset_agent()

    def run(self, reset=True) -> None:
        if reset:
            self.__reset_agent()

        facts = "Features:\n" + self.features + "\n\nCRT Prediction: "
        self.scratchpad += facts
        # print(facts, end="")

        self.scratchpad += self.prompt_agent()
        response = self.scratchpad.split('CRT Prediction: ')[-1]
        self.prediction = response.split('\n')[0].strip()
        # print(response, end="\n\n\n\n")

        self.finished = True
    
    def agent_run(self, order, reset=True):
        if reset:
            self.__reset_agent()
        facts = "Features:\n" + self.features + "\n\nCRT Prediction: "
        self.scratchpad += facts
        # print(facts, end="")

        agent_predict_prompt = self.agent_prompt.format(
                            examples = self.predict_examples,
                            order = order,
                            features = self.features)
        self.scratchpad += self.agent_llm(agent_predict_prompt)
        response = self.scratchpad.split('CRT Prediction: ')[-1]
        self.prediction = response.split('\n')[0].strip()

        self.finished = True

    def guide_run(self, rules, reset=True) -> None:
        if reset:
            self.__reset_agent()
        facts = "Features:\n" + self.features + "\n\nCRT Prediction: "
        self.scratchpad += facts
        # print(facts, end="")

        guide_predict_prompt = self.rule_prompt.format(
                            examples = self.predict_examples,
                            rules = rules,
                            features = self.features)
        self.scratchpad += self.predict_llm(guide_predict_prompt)
        response = self.scratchpad.split('CRT Prediction: ')[-1]
        self.prediction = response.split('\n')[0].strip()

        self.finished = True

    def prompt_agent(self) -> str:
        return self.llm(self._build_agent_prompt())

    def _build_agent_prompt(self) -> str:
        return self.predict_prompt.format(
                            examples = self.predict_examples,
                            features = self.features)

    def is_finished(self) -> bool:
        return self.finished

    def is_correct(self) -> bool:
        return EM(self.target, self.prediction)

    def __reset_agent(self) -> None:
        self.finished = False
        self.scratchpad: str = ''


def EM(target, prediction) -> bool:
    return prediction.lower() == target.lower().strip()
