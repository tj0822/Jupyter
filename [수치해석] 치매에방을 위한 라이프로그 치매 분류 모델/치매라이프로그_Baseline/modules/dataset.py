"""
"""

import os

import numpy as np
import pandas as pd
import torch
import modules.utils as utils
from torch.utils.data import Dataset
from sklearn import preprocessing
import sys


class CustomDataset(Dataset):
    """
    """
    def __init__(self, data_dir, mode):
        self.data_dir = data_dir
        self.mode = mode
        self.states = {'CN': 0, 'MCI': 1, 'Dem': 2}
        self.inputs, self.labels = self.data_loader()

    def data_loader(self):
        print('Loading ' + self.mode + ' dataset..')
        if not os.path.isdir(self.data_dir):
            print(f'!!! Cannot find {self.data_dir}... !!!')
            sys.exit()

        if os.path.isfile(os.path.join(self.data_dir, self.mode, self.mode + '_X.pt')):
            inputs = torch.load(os.path.join(self.data_dir, self.mode, self.mode + '_X.pt'))
            labels = torch.load(os.path.join(self.data_dir, self.mode, self.mode + '_Y.pt'))

        else:
            inputs, labels = utils.load_csv(os.path.join(self.data_dir, self.mode, self.mode + '.csv')), utils.load_csv(os.path.join(self.data_dir, self.mode, self.mode + '_label.csv'))
            inputs, labels = self.preprocessing(inputs, labels)
            torch.save(inputs, os.path.join(self.data_dir, self.mode, self.mode + '_X.pt'))
            torch.save(labels, os.path.join(self.data_dir, self.mode, self.mode + '_Y.pt'))

        return inputs, labels

    # def preprocessing(self, inputs, labels):
    def preprocessing(self, inputs, labels):
        print('Preprocessing ' + self.mode + ' dataset..')
        # Cut time series length based on the shortest length
        if self.mode == 'train':
            val_df = utils.load_csv(os.path.join(self.data_dir, 'val', 'val.csv'))
            test_df = utils.load_csv(os.path.join(self.data_dir, 'test', 'test.csv'))
            time_series_length = pd.concat([inputs['EMAIL'].value_counts(), val_df['EMAIL'].value_counts(), test_df['EMAIL'].value_counts()])
        elif self.mode == 'val':
            train_df = utils.load_csv(os.path.join(self.data_dir, 'train', 'train.csv'))
            test_df = utils.load_csv(os.path.join(self.data_dir, 'test', 'test.csv'))
            time_series_length = pd.concat([inputs['EMAIL'].value_counts(), train_df['EMAIL'].value_counts(), test_df['EMAIL'].value_counts()])
        else:
            train_df = utils.load_csv(os.path.join(self.data_dir, 'train', 'train.csv'))
            val_df = utils.load_csv(os.path.join(self.data_dir, 'val', 'val.csv'))
            time_series_length = pd.concat([inputs['EMAIL'].value_counts(), train_df['EMAIL'].value_counts(), val_df['EMAIL'].value_counts()])

        shortest_length = time_series_length[-1]
        arranged_labels = []

        for id in inputs['EMAIL'].unique():
            idx = inputs['EMAIL'][inputs['EMAIL'] == id].index
            start_idx = idx[0]
            end_idx = idx[-1]
            inputs.drop(list((range(start_idx + shortest_length , end_idx+1))), axis=0, inplace=True)
            inputs = inputs.reset_index(drop=True)
            label_idx = labels['SAMPLE_EMAIL'][labels['SAMPLE_EMAIL'] == id].index[0]
            arranged_labels.append(labels['DIAG_NM'][label_idx])

        # Encoding label
        labels = torch.tensor(self.label_encoder(arranged_labels))

        # Selecting usage columns
        del_col = [ 'EMAIL', 'summary_date',
                   'activity_class_5min', 'activity_met_1min',
                   'sleep_hr_5min', 'sleep_hypnogram_5min', 'sleep_rmssd_5min', 'timezone', 'sleep_total',
                   'CONVERT(activity_class_5min USING utf8)', 'CONVERT(activity_met_1min USING utf8)',
                   'CONVERT(sleep_hr_5min USING utf8)', 'CONVERT(sleep_hypnogram_5min USING utf8)',
                   'CONVERT(sleep_rmssd_5min USING utf8)']
        if self.mode == 'val':
            inputs.drop(del_col[1:], axis=1, inplace=True)
        else:
            inputs.drop(del_col, axis=1, inplace=True)


        #Normalization
        scaler = preprocessing.StandardScaler()
        if self.mode == 'test' :
            train_df.drop(del_col, axis=1, inplace=True)
            scaler.fit_transform(train_df)
            inputs = scaler.transform(inputs)
        else:
            inputs = scaler.fit_transform(inputs)

        # Convert dataframe to tensor
        inputs = torch.FloatTensor(inputs).reshape(len(labels), -1, inputs.shape[1])
        labels = torch.LongTensor(labels)

        return inputs, labels

    def label_encoder(self, labels):
        try:
            labels = list(map(lambda x : self.states[x], labels))
            return labels
        except:
            assert 'Invalid states'

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index, :, :], self.labels[index]




