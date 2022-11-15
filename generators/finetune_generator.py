import os
import pickle
import pandas as pd
from generators.generator import Generator
from generators.data import EHRTokenizer, FinetuneEHRDataset


class FinetuneGenerator(Generator):

    def __init__(self, args, logger, device):

        self.hos_index = None
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
            if self.args.single_train:   # single train, single test
                data_list = self._load_single_data(data_dir, self.args.hos_id)
                #data_list = self._split_dataset(data)

            elif self.args.single_test:  # full train, single test
                #data = self._load_single_data(data_dir, self.args.hos_id)
                #data_list = self._split_dataset(data) # split dataset based on hos id data
                #train_data = self._load_multi_data(data_dir, self.args.hos_id)
                #data_list[0] = pd.concat([data_list[0], train_data])    # concat other data and hos id train data
                data_list = self._load_multi_data(data_dir, hos_list)
                self.hos_index = data_list[2].hospital_id.values

            else:   # full train, full test
                record_dir = os.path.join(data_dir, 'data-single-visit.raw.pkl')
                # load data
                data = pd.read_pickle(record_dir)
                # load data_list = [trian, eval, test data]
                data_list = self._split_dataset(data)

        # load tokenizer
        self.tokenizer = EHRTokenizer(voc_dir)
        
        self.train_dataset = FinetuneEHRDataset(data_list[0], self.tokenizer, max_seq_len)
        self.eval_dataset = FinetuneEHRDataset(data_list[1], self.tokenizer, max_seq_len)
        self.test_dataset = FinetuneEHRDataset(data_list[2], self.tokenizer, max_seq_len)


    def _load_single_data_old(self, data_dir, hos_id=None):
        # only load the hos_id data
        data_dir = os.path.join(data_dir, 'data-single-visit-multi-center.raw.pkl')
        data_dict = pickle.load(open(data_dir, 'rb'))

        try:
            return data_dict[hos_id]
        except:
            raise ValueError("Please input the correct hospital id.")


    def _load_multi_data_old(self, data_dir, hos_id=None):
        # load all data, except for the hos in hos_id
        data_dir = os.path.join(data_dir, 'data-single-visit-multi-center.raw.pkl')
        data_dict = pickle.load(open(data_dir, 'rb'))
        df_list = []

        for key in data_dict.keys():
            if key == hos_id:   # don't load the hos id data
                continue
            df_list.append(data_dict[key])

        return pd.concat(df_list)




