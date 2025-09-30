from data_load.dataloader import DataLoader
from utils.util import summarize_trial, remove_reflections, save_results, confusion_matrix
from utils.agents import PredictAgent
from utils.llm import OpenAILLM,PipeLLM,AgentLLM
from utils.crt_prompts import MODIFY_GUIDELINE_INSTRUCTION, SUMMARIZE_REVISE_INSTRUCTION
import os, json
import logging
import openai
from tqdm import tqdm
import pickle

class Exp_Model:
    def __init__(self, args):
        self.args = args
        self.dataloader = DataLoader(args)
        self.train_data,self.test_data = self.dataloader.load()
        print(self.test_data['target'].value_counts())
        print(self.train_data['target'].value_counts())
        
        # checkpoint directory
        parentdir, _ = os.path.split(self.args.save_dir)
        save_path = os.path.join(parentdir, PipeLLM().model.split('/')[-1],_)
        print(save_path)
        self.checkpoint_dir = os.path.join(save_path, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.latest_checkpoint = os.path.join(self.checkpoint_dir, "latest_checkpoint.pkl")
        
        self.epoch_checkpoint_pattern = os.path.join(self.checkpoint_dir, "checkpoint_epoch_{}.pkl")

    def _save_checkpoint(self, state, epoch):
        """save checkpoint state (always overwrite the latest checkpoint file)"""
        epoch_checkpoint_path = self.epoch_checkpoint_pattern.format(epoch)
        with open(epoch_checkpoint_path, 'wb') as f:
            pickle.dump(state, f)
        
        with open(self.latest_checkpoint, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"Save checkpoint to: {epoch_checkpoint_path}")
        print(f"Update latest checkpoint: {self.latest_checkpoint}")

    def _load_checkpoint(self):
        """load checkpoint if exists"""
        if not os.path.exists(self.latest_checkpoint):
            print("No checkpoint found, start from scratch.")
            return None
        try:
            with open(self.latest_checkpoint, 'rb') as f:
                state = pickle.load(f)
            return state
        except Exception as e:
            print(f"Loading checkpoint failed: {e}")
            return None

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
        num_per_group = self.args.group_size
        Guide_LLM = PipeLLM()
        Reflect_LLM = PipeLLM()
        parentdir, _ = os.path.split(self.args.save_dir)
        save_path = os.path.join(parentdir, PipeLLM().model.split('/')[-1],_)
        os.makedirs(save_path, exist_ok=True)
        case_study_path = os.path.join(save_path, 'case_study.json')
        with open(case_study_path, 'w') as f:
            f.write('')
        extract_task = self.args.feature_dir.split('/')[-1].strip()
        cur_guideline_name = 'current_guideline.json'
        best_guideline_name = 'guideline.json'
        logging_name = 'logging.txt'
        explanation_name = 'explanation.json'
        distribution_name = 'distribution.json'
        logging_path = os.path.join(save_path, logging_name)
        with open(logging_path, 'w') as f:
            f.write("")
        logging.basicConfig(filename=logging_path, level=logging.WARNING,
                    format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger("my_logger")
        logger.setLevel(logging.INFO)
        logger.info(extract_task)
        current_guideline_path = os.path.join(save_path, cur_guideline_name)
        guideline_path = os.path.join(save_path, best_guideline_name)
        explanation_path = os.path.join(save_path, explanation_name)
        distribution_path = os.path.join(save_path, distribution_name)

        # initilize or load training state
        current_epoch = 0  
        best_mcc = -1.0
        guideline = ""
        best_guideline = ""
        distribution = ""
        wrong_agents = []
        reflect_samples = ''
        explanations = []
        iterative_predictions = []
        iterative_list = []
        reflect_prompt = MODIFY_GUIDELINE_INSTRUCTION
        summarize_prompt = SUMMARIZE_REVISE_INSTRUCTION

        # load ckpt if exists
        checkpoint = self._load_checkpoint()
        if self.args.load_ckpt:
            try:
                # recover training state
                current_epoch = checkpoint['current_epoch']
                best_mcc = checkpoint['best_mcc']
                guideline = checkpoint['guideline']
                best_guideline = checkpoint['best_guideline']
                distribution = checkpoint['distribution']
                wrong_agents = checkpoint['wrong_agents']
                reflect_samples = checkpoint['reflect_samples']
                explanations = checkpoint['explanations']
                iterative_predictions = checkpoint['iterative_predictions']
                iterative_list = checkpoint['iterative_list']
                agents = checkpoint['agents']
                valid_agents = checkpoint['valid_agents']
                print(f"Loaded latest checkpoint, continue training from epoch {checkpoint['current_epoch'] + 1} ")

                with open(distribution_path, 'w') as f:
                    f.write(distribution+'\n')
            except Exception as e:
                print(f"checkpoint damaged or incomplete: {e}, start from scratch.")
                current_epoch = 0
                best_mcc = -1.0
                guideline = ""
                best_guideline = ""
                distribution = ""
                wrong_agents = []
                reflect_samples = ''
                explanations = []
                iterative_predictions = []
                iterative_list = []
            
        
        with open(distribution_path, 'r') as f:
            distribution = f.read().strip()
        
        # self-reflection process
        for trial in range(current_epoch, self.args.num_epochs):
            
            if trial > current_epoch:
                for agent in agents:
                    agent.is_trained = False
                for agent in valid_agents:
                    agent.is_validated = False

            # training
            for idx, agent in enumerate(tqdm(agents, desc=f"Training Agents in Trial {trial + 1}")):
                if agent.is_trained:  
                    continue
                try:
                    agent.guide_run(guideline, distribution)
                    agent.is_trained = True  
                except Exception as e:
                    print(f"Encountered API timeout {e} while training agent {idx}. Saving breakpoint and exiting.")
                    self._save_checkpoint({
                        'current_epoch': trial,
                        'best_mcc': best_mcc,
                        'guideline': guideline,
                        'best_guideline': best_guideline,
                        'distribution': distribution,
                        'wrong_agents': wrong_agents,
                        'reflect_samples': reflect_samples,
                        'explanations': explanations,
                        'iterative_predictions': iterative_predictions,
                        'iterative_list': iterative_list,
                        'agents': agents,
                        'valid_agents': valid_agents
                    }, trial)
                    return

                # reflection
                if not agent.is_correct():
                    wrong_agents.append(agent)
                    reflect_samples += f'sample{len(wrong_agents)}: {agent.features}\n true Disease label: {agent.target}\n'
                    if len(wrong_agents) == num_per_group:    
                        logger.info(f"Epoch {trial + 1} rule reflection")                    
                        prompt = reflect_prompt.format(
                            distribution=distribution,
                            rules=guideline,
                            samples=reflect_samples
                        )
                        try:
                            response = Reflect_LLM(prompt)
                        except Exception as e:
                            print(f"Encountered API timeout {e} while reflection. Saving breakpoint and exiting.")
                            self._save_checkpoint({
                                'current_epoch': trial,
                                'best_mcc': best_mcc,
                                'guideline': guideline,
                                'best_guideline': best_guideline,
                                'distribution': distribution,
                                'wrong_agents': wrong_agents,
                                'reflect_samples': reflect_samples,
                                'explanations': explanations,
                                'iterative_predictions': iterative_predictions,
                                'iterative_list': iterative_list,
                                'agents': agents,
                                'valid_agents': valid_agents
                            }, trial)
                            return
                        guideline = response.split("Rules:")[-1].strip()
                        reflect_samples = ''
                        wrong_agents.clear()

            # save checkpoint after training phase
            self._save_checkpoint({
                'current_epoch': trial,
                'best_mcc': best_mcc,
                'guideline': guideline,
                'best_guideline': best_guideline,
                'distribution': distribution,
                'wrong_agents': wrong_agents,
                'reflect_samples': reflect_samples,
                'explanations': explanations,
                'iterative_predictions': iterative_predictions,
                'iterative_list': iterative_list,
                'agents': agents,
                'valid_agents': valid_agents
            }, trial)

            # log training results
            correct, incorrect = summarize_trial(agents)
            print(f'Finished Training {trial + 1}, Correct: {len(correct)}, Incorrect: {len(incorrect)}')
            logger.info(f'Finished Training {trial + 1}, Correct: {len(correct)}, Incorrect: {len(incorrect)}')

            # rules check
            try:
                prompt = summarize_prompt.format(distribution=distribution, rules=guideline)
                response = Reflect_LLM(prompt)
            except Exception as e:
                print(f"Encountered API timeout {e} while check rules. Saving breakpoint and exiting.")
                self._save_checkpoint({
                    'current_epoch': trial,
                    'best_mcc': best_mcc,
                    'guideline': guideline,
                    'best_guideline': best_guideline,
                    'distribution': distribution,
                    'wrong_agents': wrong_agents,
                    'reflect_samples': reflect_samples,
                    'explanations': explanations,
                    'iterative_predictions': iterative_predictions,
                    'iterative_list': iterative_list,
                    'agents': agents,
                    'valid_agents': valid_agents
                }, trial)
                return
            guideline = response.split("Rules:")[-1].strip()
            logger.info(f"Epoch {trial + 1} summarized guideline: {guideline}")

            # validation
            for ind, agent in tqdm(enumerate(valid_agents), total=len(valid_agents), desc=f"Validation Agents in Trial {trial + 1}"):
                if agent.is_validated:  
                    continue
                try:
                    agent.guide_run(guideline, distribution)
                    agent.is_validated = True
                except Exception as e:
                    print(f"Encountered API timeout {e} while validating agent {ind}. Saving breakpoint and exiting.")
                    self._save_checkpoint({
                        'current_epoch': trial,
                        'best_mcc': best_mcc,
                        'guideline': guideline,
                        'best_guideline': best_guideline,
                        'distribution': distribution,
                        'wrong_agents': wrong_agents,
                        'reflect_samples': reflect_samples,
                        'explanations': explanations,
                        'iterative_predictions': iterative_predictions,
                        'iterative_list': iterative_list,
                        'agents': agents,
                        'valid_agents': valid_agents
                    }, trial)
                    return

                if ind in iterative_list:
                    response = agent.scratchpad.split('Disease Prediction:')[-1].strip()
                    sample = {"epoch":trial,"features": agent.features, "label": agent.target, "output": response}
                    iterative_predictions.append(sample)
                
                response = agent.scratchpad.split('Disease Prediction:')[-1].strip()
                sample = {"features": agent.features, "label": agent.prediction, "output": response}
                explanations.append(sample)

            # save results
            with open(explanation_path,'w') as f:
                json.dump(explanations,f)
            explanations = []

            # update best metrics
            correct, incorrect = summarize_trial(valid_agents)
            logger.info(f'Finished Validation {trial + 1}, Correct: {len(correct)}, Incorrect: {len(incorrect)}')
            print(f'Finished Validation {trial + 1}, Correct: {len(correct)}, Incorrect: {len(incorrect)}')
            tp, tn, fp, fn = confusion_matrix(correct, incorrect)
            mcc = (tp*tn - fp*fn) / ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**0.5 if (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) > 0 else 0
            print(f'MCC:{mcc}')
            if mcc >= best_mcc:
                best_guideline = guideline
                best_mcc = mcc
                logger.info(f'Trial {trial+1} reached best correct {best_mcc}, saved.')
                with open(guideline_path, 'w') as f:
                    f.write(best_guideline + "\n")

            # save checkpoint after validation phase
            self._save_checkpoint({
                'current_epoch': trial,  
                'best_mcc': best_mcc,
                'guideline': guideline,
                'best_guideline': best_guideline,
                'distribution': distribution,
                'wrong_agents': wrong_agents,
                'reflect_samples': reflect_samples,
                'explanations': explanations,
                'iterative_predictions': iterative_predictions,
                'iterative_list': iterative_list,
                'agents': agents,
                'valid_agents': valid_agents
            }, trial)

            # save current guideline
            with open(current_guideline_path, 'w') as f:
                f.write(guideline+'\n')
            with open(case_study_path, 'w') as f:
                json.dump(iterative_predictions,f)

        print("Finished training.")



    def test(self):
        print("Loading Test Agents...")
        data = self.test_data

        agent_cls = PredictAgent
        test_agents = [agent_cls(row['ticker'], row['features'], row['target']) for _, row in data.iterrows()]
        print("Loaded Test Agents.")
        logger = logging.getLogger("my_logger")
        logger.setLevel(logging.INFO)
        explanations = []
        extract_task = self.args.feature_dir.split('/')[-1].strip()
        explanation_name = 'results.json'
        best_guideline_name = 'guideline.json'
        parentdir, _ = os.path.split(self.args.save_dir)
        save_path = os.path.join(parentdir, PipeLLM().model.split('/')[-1],_)
        guideline_path = os.path.join(save_path, best_guideline_name)
        explanation_path = os.path.join(save_path, explanation_name)
        distribution_path = os.path.join(save_path, 'distribution.json')

        # load the guidelines
        with open(guideline_path, 'r') as f:
            guidelines = f.read()
        with open(distribution_path, 'r') as f:
            distribution = f.read().strip()

        for agent in tqdm(test_agents, desc="Testing Agents"):
            agent.guide_run(guidelines, distribution)
            
            response = agent.scratchpad.split('Disease Prediction:')[-1].strip()
            sample = {"features": agent.features, "label": agent.target, "output": response}
            explanations.append(sample)

        with open(explanation_path,'w') as f:
            json.dump(explanations,f)

        correct, incorrect = summarize_trial(test_agents)
        tp, tn, fp, fn = confusion_matrix(correct, incorrect)
        print(f'Finished evaluation, Correct: {len(correct)}, Incorrect: {len(incorrect)}, Accuracy:{len(correct)/(len(correct)+len(incorrect))}')
        print(f'TP:{tp},TN:{tn},FP:{fp},FN:{fn}')
        print(f'MCC:{(tp*tn-fp*fn)/((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**0.5}')
        precision = tp/(tp+fp) if (tp+fp)>0 else 0
        recall = tp/(tp+fn) if (tp+fn)>0 else 0
        f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0
        print(f'Precision:{precision},Recall:{recall},F1:{f1}')
        logger.info(f'Finished evaluation, Correct: {len(correct)}, Incorrect: {len(incorrect)}')
        logger.info(f'TP:{tp},TN:{tn},FP:{fp},FN:{fn}')
        logger.info(f'MCC:{(tp*tn-fp*fn)/((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**0.5}')
        logger.info(f'Precision:{precision},Recall:{recall},F1:{f1}')

