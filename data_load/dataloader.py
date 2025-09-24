import os, json
import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta


class DataLoader:
    def __init__(self, args):
        self.label_dir = args.label_dir
        self.feature_dir = args.feature_dir


    def daterange(self, start_date, end_date):
        for n in range(int((end_date - start_date).days)):
            yield start_date + timedelta(n)




    def get_disease(self, disease_path):
        if os.path.exists(disease_path):
            with open(disease_path) as f:
                lines = f.readlines()
            disease = [line.strip() for line in lines]
        return disease

    def get_features(self, text_path):
        features = []

        if os.path.exists(text_path):
            with open(text_path) as f:
                lines = f.readlines()
                for line in lines:
                    tweet_obj = json.loads(line)
                    features.append(tweet_obj['text'])
        return features


    def load(self):
        train_data = pd.DataFrame()
        test_data = pd.DataFrame()

        labels = self.get_disease(self.label_dir)
        features = self.get_features(self.feature_dir)
        ticker = self.feature_dir.split('/')[-1]

        datalen = len(features)
        train_idx = round(datalen*0.8)

        for i in range(train_idx):
            train_data = pd.concat([train_data, pd.DataFrame([{'ticker': ticker, 'target': labels[i], 'features': features[i]}])], ignore_index=True)
        for i in range(train_idx,datalen):
            test_data = pd.concat([test_data, pd.DataFrame([{'ticker': ticker, 'target': labels[i], 'features': features[i]}])], ignore_index=True)
        return train_data,test_data
