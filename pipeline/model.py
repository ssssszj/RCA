from data_load.dataloader import DataLoader
from utils.util import summarize_trial, remove_reflections, save_results, confusion_matrix
from utils.agents import PredictAgent
from utils.llm import OpenAILLM,PipeLLM,AgentLLM
from utils.prompts import ORIGIN_GUIDELINE_INSTRUCTION,MODIFY_GUIDELINE_INSTRUCTION,AGENT_PREDICT_INSTRUCTION
import os, json
import logging
from tqdm import tqdm


class Exp_Model:
    def __init__(self, args):
        self.args = args
        self.dataloader = DataLoader(args)
        self.train_data,self.test_data = self.dataloader.load()
        print(self.test_data['target'].value_counts())
        print(self.train_data['target'].value_counts())

    def train(self):
        # Collect demonstration data
        print("Loading Train Agents...")
        data = self.train_data

        agent_cls = PredictAgent    
        agents = [agent_cls(row['ticker'], row['features'], row['target']) for _, row in data.iterrows()]
        print("Loaded Train Agents.")

        # initialization
        length = len(agents)
        valid_agents = agents[length//4*3:-1]
        agents = agents[:length//4*3]
        num_per_group = 30
        Guide_LLM = PipeLLM()
        Reflect_LLM = PipeLLM()


        with open('logging.txt', 'w') as f:
            f.write("")
        logging.basicConfig(filename='logging.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
        os.makedirs(self.args.datasets_dir, exist_ok=True)
        wrong_predictions_path = os.path.join(self.args.datasets_dir, "wrong_predictions.json")
        current_guideline_path = os.path.join(self.args.datasets_dir, "current_guideline.json")
        guideline_path = os.path.join(self.args.datasets_dir, "guideline.json")
        order_path = os.path.join(self.args.datasets_dir, "order.json")
        with open(order_path, 'r') as f:
            order = f.read()

        # generate original guidelines for afterwards reflection
        '''
        guide_prompt=ORIGIN_GUIDELINE_INSTRUCTION
        original_samples = ""
        for i in range(num_per_group):
            original_samples += f'sample{i+1}: {agents[i].features}\n CRT result: {agents[i].target}\n'
        prompt = guide_prompt.format(
            samples=original_samples
        )
        response = Guide_LLM(prompt)
        guideline = response.split("Rules: ")[-1]
        
        print(guideline)
        logging.info(f"Original guideline: {guideline}")
        with open(current_guideline_path, 'w') as f:
            f.write(guideline+'\n')

        '''
        # 继续训练 加载当前规则
        with open('./CRT_data/current_guideline.json', 'r') as f:
            guideline = f.read()
        

        # reflection on rules

        wrong_agents = []
        reflect_samples = ''
        wrong_predictions = []
        
        # 继续训练 加载最佳规则
        with open(guideline_path, 'r') as f:
            best_guideline = f.read()
        '''
        # guideline = best_guideline #to retrain the guideline from best checkpoint
        
        best_guideline = guideline
        
        for agent in tqdm(valid_agents, desc=f"Validation Agents in Trial 0"):
            agent.guide_run(guideline)

        correct, incorrect = summarize_trial(valid_agents)
        print(f'Finished Validation 0, Correct: {len(correct)}, Incorrect: {len(incorrect)}')
        logging.info(f'Finished Validation 0, Correct: {len(correct)}, Incorrect: {len(incorrect)}')
        with open(guideline_path, 'w') as f:
            f.write(best_guideline + "\n")
        '''
        best_correct = 48 #len(correct)
        reflect_prompt = MODIFY_GUIDELINE_INSTRUCTION

        # 使用 tqdm 包裹外层循环，显示试验进度
        for trial in range(self.args.num_epochs):
            # train
            for agent in tqdm(agents, desc=f"Training Agents in Trial {trial + 1}"):
                agent.guide_run(guideline)
                # print(agent.is_correct())

                if agent.is_correct():
                    # print(agent.scratchpad)
                    order = agent.scratchpad.split('Feature Importance Ordering:')[-1]
                    # print(order)
                    order = order.strip()
                    with open(order_path, 'w') as f:
                        f.write(order+'\n')

                if not agent.is_correct():
                    wrong_agents.append(agent)
                    reflect_samples += f'sample{len(wrong_agents)}: {agent.features}\n true CRT result: {agent.target}\n'
                    prompt = remove_reflections(agent._build_agent_prompt())
                    response = agent.scratchpad.split('CRT Prediction: ')[-1]
                    sample = {"user_input": prompt, "label": agent.target, "output": response}
                    if len(wrong_agents) == num_per_group:
                        
                        prompt = reflect_prompt.format(
                            guidelines=guideline,
                            samples=reflect_samples
                        )
                        response = Reflect_LLM(prompt)
                        # print(response)
                        guideline = response.split("Rules: ")[-1]
                        reflect_samples = ''
                        wrong_agents.clear()

            correct, incorrect = summarize_trial(agents)
            print(f'Finished Trial {trial + 1}, Correct: {len(correct)}, Incorrect: {len(incorrect)}')
            logging.info(f'Finished Trial {trial + 1}, Correct: {len(correct)}, Incorrect: {len(incorrect)}')

            # validate
            wrong_predictions.clear()
            for agent in tqdm(valid_agents, desc=f"Validation Agents in Trial {trial + 1}"):
                agent.guide_run(guideline)

                if not agent.is_correct():
                    prompt = remove_reflections(agent._build_agent_prompt())
                    response = agent.scratchpad.split('CRT Prediction: ')[-1]
                    sample = {"user_input": prompt, "label": agent.target, "output": response}
                    wrong_predictions.append(sample)
                    
            
            correct, incorrect = summarize_trial(valid_agents)
            logging.info(f'Finished Validation {trial + 1}, Correct: {len(correct)}, Incorrect: {len(incorrect)}')
            print(f'Finished Validation {trial + 1}, Correct: {len(correct)}, Incorrect: {len(incorrect)}')
            if len(correct) > best_correct:
                best_guideline = guideline
                best_correct = len(correct)
                logging.info(f'Trial {trial+1} reached best correct {best_correct}, saved.')
                with open(guideline_path, 'w') as f:
                    f.write(best_guideline + "\n")
                with open(wrong_predictions_path,'w') as f:
                    json.dump(wrong_predictions,f)

            # temporary save
            with open(current_guideline_path, 'w') as f:
                f.write(guideline+'\n')

       
        print("Finished training.")

    def test(self):
        print("Loading Test Agents...")
        data = self.test_data

        agent_cls = PredictAgent
        test_agents = [agent_cls(row['ticker'], row['features'], row['target']) for _, row in data.iterrows()]
        print("Loaded Test Agents.")
        wrong_predictions = []
        wrong_predictions_path = os.path.join(self.args.datasets_dir, "wrong_predictions.json")
        order_path = os.path.join(self.args.datasets_dir, "order.json")
        guideline_path = os.path.join(self.args.datasets_dir, "guideline.json")

        # load the guidelines
        with open(guideline_path, 'r') as f:
            guidelines = f.read()
        with open(order_path, 'r') as f:
            order = f.read()
        for agent in tqdm(test_agents, desc="Testing Agents"):
            agent.guide_run(guidelines)

            if not agent.is_correct():
                prompt = remove_reflections(agent._build_agent_prompt())
                response = agent.scratchpad.split('CRT Prediction: ')[-1]
                sample = {"user_input": prompt, "label": agent.target, "output": response}
                wrong_predictions.append(sample)

        # with open(wrong_predictions_path,'w') as f:
        #     f.write('[\n')
        #     for i,data in enumerate(wrong_predictions):
        #         json.dump(data,f)
        #         f.write(',\n') if i != len(wrong_predictions)-1 else f.write('\n')
        #     f.write(']')

        correct, incorrect = summarize_trial(test_agents)
        tp, tn, fp, fn = confusion_matrix(correct, incorrect)
        print(f'Finished evaluation, Correct: {len(correct)}, Incorrect: {len(incorrect)}, Accuracy:{len(correct)/(len(correct)+len(incorrect))}')
        print(f'TP:{tp},TN:{tn},FP:{fp},FN:{fn}')
        print(f'MCC:{(tp*tn-fp*fn)/((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**0.5}')
        logging.info(f'Finished evaluation, Correct: {len(correct)}, Incorrect: {len(incorrect)}')

        save_results(test_agents, self.args.save_dir)
