import os
import time
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


class Generator(object):

    def __init__(self, args, logger, device):

        self.args = args
        self.num_workers = args.num_workers
        self.bs = args.train_batch_size
        self.logger = logger
        self.device = device

        self.logger.info("Loading dataset ... ")
        start = time.time()
        self._load_dataset()
        end = time.time()
        self.logger.info("Dataset is loaded: consume %.3f s" % (end - start))

    
    def _load_dataset(self):

        return NotImplementedError

    
    def _split_dataset(self, data):
        '''Split the datatset based on the ratio 8:1:1'''
        index_list = list(range(data.shape[0]))
        train_num, val_num = int(0.8 * data.shape[0]), int(0.1 * data.shape[0])

        train_index = np.random.choice(index_list, size=train_num, replace=False)
        val_index = np.random.choice(list(set(index_list) - set(train_index)), size=val_num, replace=False)
        test_index = list(set(index_list) - set(train_index) - set(val_index))

        return [data.iloc[train_index, :], data.iloc[val_index, :], data.iloc[test_index, :]]


    def _load_multi_data(self, data_dir, hos_list):
        # load all data
        
        train_list, val_list, test_list = [], [], []
        data_dir = os.path.join(data_dir, str(self.args.seed))

        for hos_id in hos_list:
            data_path = os.path.join(data_dir, str(hos_id))
            train_list.append(pickle.load(open(os.path.join(data_path, 'train.pkl'), 'rb')))
            val_list.append(pickle.load(open(os.path.join(data_path, 'val.pkl'), 'rb')))
            test_list.append(pickle.load(open(os.path.join(data_path, 'test.pkl'), 'rb')))

        return pd.concat(train_list), pd.concat(val_list), pd.concat(test_list)

    
    def _load_single_data(self, data_dir, hos_id):
        '''Load the data of specified hos id'''

        try:
            data_dir = os.path.join(data_dir, str(self.args.seed))
            data_path = os.path.join(data_dir, str(hos_id))
            train = pickle.load(open(os.path.join(data_path, 'train.pkl'), 'rb'))
            val = pickle.load(open(os.path.join(data_path, 'val.pkl'), 'rb'))
            test = pickle.load(open(os.path.join(data_path, 'test.pkl'), 'rb'))

            return train, val, test

        except:
            raise ValueError('Please enter the correct hospital id')


    def make_dataloaders(self):

        train_dataloader = DataLoader(self.train_dataset,
                                      sampler=RandomSampler(self.train_dataset),
                                      batch_size=self.bs,
                                      num_workers=self.num_workers)
        eval_dataloader = DataLoader(self.eval_dataset,
                                     sampler=SequentialSampler(self.eval_dataset),
                                     batch_size=100,
                                     num_workers=self.num_workers)
        test_dataloader = DataLoader(self.test_dataset,
                                     sampler=SequentialSampler(self.test_dataset),
                                     batch_size=100,
                                     num_workers=self.num_workers)

        return train_dataloader, eval_dataloader, test_dataloader
    

    def get_tokenizer(self):

        if self.tokenizer:

            return self.tokenizer

        else:

            raise ValueError("Please initialize the generator firstly")

    
    def get_statistics(self):

        return len(self.train_dataset), len(self.eval_dataset), len(self.test_dataset)



