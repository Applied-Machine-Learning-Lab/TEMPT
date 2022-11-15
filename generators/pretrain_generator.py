import os
import pickle
import pandas as pd
from generators.generator import Generator
from generators.data import EHRTokenizer, PretrainContrastiveEHRDataset


class PretrainGenerator(Generator):

    def __init__(self, args, logger, device):
        
        super().__init__(args, logger, device)


    def _load_dataset(self):

        data_dir = self.args.data_dir
        max_seq_len = self.args.max_seq_length
        hos_list = pd.read_csv(os.path.join(data_dir, 'hospital.csv')).hospital_id.values
        hos_list = list(hos_list)

        # whether run demo
        if self.args.demo:
            voc_dir = os.path.join(data_dir, 'vocab.demo.pkl')
            record_dir = os.path.join(data_dir, 'data-single-visit.demo.pkl')
            data = pd.read_pickle(record_dir)
            data_list = self._split_dataset(data)
        else:
            voc_dir = os.path.join(data_dir, 'vocab.raw.pkl')
            data_list = self._load_multi_data(data_dir, hos_list)
            data = pd.concat(data_list)
            #data_list = self._split_dataset(data)

        # load tokenizer
        self.tokenizer = EHRTokenizer(voc_dir)
        self.tokenizer.attri_num = data['attri'].nunique()
        self.tokenizer.hos_num = data['hospital_id'].max()+1

        self.train_dataset = PretrainContrastiveEHRDataset(data_list[0], self.tokenizer, max_seq_len)
        self.eval_dataset = PretrainContrastiveEHRDataset(data_list[1], self.tokenizer, max_seq_len)
        self.test_dataset = PretrainContrastiveEHRDataset(data_list[2], self.tokenizer, max_seq_len)


    
    def _load_multi_data_old(self, data_dir, hos_id=None):
        # load all data, except for the hos in hos_id
        data_dir = os.path.join(data_dir, 'data-single-visit-multi-center.attri.raw.pkl')
        data_dict = pickle.load(open(data_dir, 'rb'))
        df_list = []

        for key in data_dict.keys():
            if key == hos_id:   # don't load the hos id data
                continue
            df_list.append(data_dict[key])

        return pd.concat(df_list)

    
    

