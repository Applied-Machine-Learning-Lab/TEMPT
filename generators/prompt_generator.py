import os
import pandas as pd
from generators.data import EHRTokenizer, PromptEHRDataset
from generators.finetune_generator import FinetuneGenerator


class PromptGenerator(FinetuneGenerator):

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

        self.train_dataset = PromptEHRDataset(data_list[0], self.tokenizer, max_seq_len, self.args.prompt_num)
        self.eval_dataset = PromptEHRDataset(data_list[1], self.tokenizer, max_seq_len, self.args.prompt_num)
        self.test_dataset = PromptEHRDataset(data_list[2], self.tokenizer, max_seq_len, self.args.prompt_num)








