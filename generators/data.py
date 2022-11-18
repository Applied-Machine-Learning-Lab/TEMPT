import numpy as np
import pandas as pd
import pickle
import copy
import os
import random

import torch
from torch.utils.data import Dataset


class Voc(object):
    '''Define the vocabulary (token) dict'''

    def __init__(self):

        self.idx2word = {}
        self.word2idx = {}

    def add_sentence(self, sentence):
        '''add vocabulary to dict via a list of words'''
        for word in sentence:
            if word not in self.word2idx:
                self.idx2word[len(self.word2idx)] = word
                self.word2idx[word] = len(self.word2idx)



class EHRTokenizer(object):
    """The tokenization that offers function of converting id and token"""

    def __init__(self, voc_dir, special_tokens=("[PAD]", "[CLS]", "[MASK]")):

        self.vocab = Voc()  # this is a overall Voc

        # special tokens
        self.vocab.add_sentence(special_tokens)

        self.diag_voc, self.med_voc, self.pro_voc = self.read_voc(voc_dir)
        self.vocab.add_sentence(self.diag_voc.word2idx.keys())
        self.vocab.add_sentence(self.med_voc.word2idx.keys())
        self.vocab.add_sentence(self.pro_voc.word2idx.keys())

        self.attri_num = None
        self.hos_num = None
    

    def read_voc(self, voc_dir):

        with open(voc_dir, 'rb') as f:
            
            voc_dict = pickle.load(f)
            
        return voc_dict['diag_voc'], voc_dict['med_voc'], voc_dict['pro_voc']


    def add_vocab(self, vocab_file):

        voc = self.vocab
        specific_voc = Voc()

        with open(vocab_file, 'r') as fin:
            for code in fin:
                voc.add_sentence([code.rstrip('\n')])
                specific_voc.add_sentence([code.rstrip('\n')])

        return specific_voc


    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            ids.append(self.vocab.word2idx[token])
        return ids


    def convert_ids_to_tokens(self, ids):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = []
        for i in ids:
            tokens.append(self.vocab.idx2word[i])
        return tokens



class EHRDataset(Dataset):
    '''The dataset for medication recommendation'''

    def __init__(self, data_pd, tokenizer: EHRTokenizer, max_seq_len):
        
        self.data_pd = data_pd
        self.tokenizer = tokenizer
        self.seq_len = max_seq_len  # the maximum length of a diagnosis/procedure record

        self.sample_counter = 0

        def transform_data(data):
            """
            :param data: the Dataframe of raw data
            :return: [dignosis_records<list>, procedure_records<list>, medication_records<list>]
            """
            records = []
            for _, row in data.iterrows():
                record = [list(row['icd9_code']), list(row['pro_code']), list(row['drug_id'])]
                records.append(record)
            
            return records

        self.records = transform_data(data_pd)

    def __len__(self):

        return NotImplementedError

    def __getitem__(self, item):

        return NotImplementedError
        


####################################
'''Pretrain Dataset'''
####################################

class PretrainEHRDataset(EHRDataset):

    def __init__(self, data_pd, tokenizer: EHRTokenizer, max_seq_len):
        
        super().__init__(data_pd, tokenizer, max_seq_len)

    
    def __len__(self):

        return len(self.records)

    
    def __getitem__(self, item):

        cur_id = item
        adm = copy.deepcopy(self.records[item])

        if len(adm[0]) >= self.seq_len - 1:
            adm[0] = adm[0][:self.seq_len-1]
        if len(adm[1]) >= self.seq_len - 1:
            adm[1] = adm[1][:self.seq_len-1]

        def fill_to_max(l, seq):
            while len(l) < seq:
                l.append('[PAD]')
            return l
        """y
        """
        y_dx = np.zeros(len(self.tokenizer.diag_voc.word2idx))
        y_rx = np.zeros(len(self.tokenizer.pro_voc.word2idx))
        for item in adm[0]:
            y_dx[self.tokenizer.diag_voc.word2idx[item]] = 1
        for item in adm[1]:
            y_rx[self.tokenizer.pro_voc.word2idx[item]] = 1

        """replace tokens with [MASK]
        """
        adm[0] = self.random_word(adm[0], self.tokenizer.pro_voc)
        adm[1] = self.random_word(adm[1], self.tokenizer.diag_voc)

        """extract input and output tokens
        """
        #random_word
        input_tokens = []  # (2*max_len)
        input_tokens.extend(
            ['[CLS]'] + fill_to_max(list(adm[0]), self.seq_len - 1))
        input_tokens.extend(
            ['[CLS]'] + fill_to_max(list(adm[1]), self.seq_len - 1))

        """convert tokens to id
        """
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

        cur_tensors = (torch.tensor(np.array(input_ids), dtype=torch.long).view(-1, self.seq_len),
                       torch.tensor(np.array(y_dx), dtype=torch.float),
                       torch.tensor(np.array(y_rx), dtype=torch.float))

        return cur_tensors

    
    def random_word(self, tokens, vocab):
        
        for i, _ in enumerate(tokens):
            prob = random.random()
            # mask token with 15% probability
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = "[MASK]"
                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.choice(list(vocab.word2idx.items()))[0]
                else:
                    pass
            else:
                pass

        return tokens



class PretrainContrastiveEHRDataset(EHRDataset):

    def __init__(self, data_pd, tokenizer: EHRTokenizer, max_seq_len):
        
        super().__init__(data_pd, tokenizer, max_seq_len)

    
    def __len__(self):

        return len(self.records)

    
    def __getitem__(self, item):

        cur_id = item
        adm = copy.deepcopy(self.records[item])

        if len(adm[0]) >= self.seq_len - 1:
            adm[0] = adm[0][:self.seq_len-1]
        if len(adm[1]) >= self.seq_len - 1:
            adm[1] = adm[1][:self.seq_len-1]

        def fill_to_max(l, seq):
            while len(l) < seq:
                l.append('[PAD]')
            return l
        """y
        """
        y_dx = np.zeros(len(self.tokenizer.diag_voc.word2idx))
        y_rx = np.zeros(len(self.tokenizer.pro_voc.word2idx))
        for item in adm[0]:
            y_dx[self.tokenizer.diag_voc.word2idx[item]] = 1
        for item in adm[1]:
            y_rx[self.tokenizer.pro_voc.word2idx[item]] = 1

        """replace tokens with [MASK]
        """
        #adm_original_0 = copy.deepcopy(adm[0])
        #adm_original_1 = copy.deepcopy(adm[1])
        adm_original_0 = adm[0]
        adm_original_1 = adm[1]
        adm[0] = self.random_word(adm[0], self.tokenizer.pro_voc)
        adm[1] = self.random_word(adm[1], self.tokenizer.diag_voc)

        """extract input and output tokens
        """
        #random_word
        input_tokens = []  # (2*max_len)
        input_tokens.extend(
            ['[CLS]'] + fill_to_max(list(adm[0]), self.seq_len - 1))
        input_tokens.extend(
            ['[CLS]'] + fill_to_max(list(adm[1]), self.seq_len - 1))

        input_tokens_original = []
        input_tokens_original.extend(
            ['[CLS]'] + fill_to_max(list(adm_original_0), self.seq_len - 1))
        input_tokens_original.extend(
            ['[CLS]'] + fill_to_max(list(adm_original_1), self.seq_len - 1))


        """convert tokens to id
        """
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
        input_ids_original = self.tokenizer.convert_tokens_to_ids(input_tokens_original)

        cur_tensors = (torch.tensor(np.array(input_ids), dtype=torch.long).view(-1, self.seq_len),
                       torch.tensor(np.array(y_dx), dtype=torch.float),
                       torch.tensor(np.array(y_rx), dtype=torch.float),
                       torch.tensor(np.array(input_ids_original), dtype=torch.long).view(-1, self.seq_len))

        return cur_tensors

    
    def random_word(self, tokens, vocab):
        
        for i, _ in enumerate(tokens):
            prob = random.random()
            # mask token with 15% probability
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = "[MASK]"
                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.choice(list(vocab.word2idx.items()))[0]
                else:
                    pass
            else:
                pass

        return tokens



####################################
'''Finetune Dataset'''
####################################

class FinetuneEHRDataset(EHRDataset):

    def __init__(self, data_pd, tokenizer: EHRTokenizer, max_seq_len):
        
        super().__init__(data_pd, tokenizer, max_seq_len)


    def __len__(self):

        return len(self.records)

    
    def __getitem__(self, item):

        cur_id = self.sample_counter
        self.sample_counter += 1
        adm = copy.deepcopy(self.records[item])

        # cut the too long records 
        if len(adm[0]) >= self.seq_len - 1:
            adm[0] = adm[0][:self.seq_len-1]
        if len(adm[1]) >= self.seq_len - 1:
            adm[1] = adm[1][:self.seq_len-1]

        def fill_to_max(l, seq):
            while len(l) < seq:
                l.append('[PAD]')
            return l

        """extract input and output tokens
        """
        input_tokens = []  # (2*max_len*adm)
        output_dx_tokens = []  # (adm-1, l)
        output_rx_tokens = []  # (adm-1, l)

        input_tokens.extend(['[CLS]'] + fill_to_max(list(adm[0]), self.seq_len - 1))
        input_tokens.extend(['[CLS]'] + fill_to_max(list(adm[1]), self.seq_len - 1))
        output_rx_tokens.append(list(adm[2]))

        #output_rx_tokens.append(list(adm[1]))
        #output_dx_tokens.append(list(adm[0]))

        """convert tokens to id
        """
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
        output_dx_labels = []  # (adm-1, dx_voc_size)
        output_rx_labels = []  # (adm-1, rx_voc_size)

        rx_voc_size = len(self.tokenizer.med_voc.word2idx)

        for tokens in output_rx_tokens:
            tmp_labels = np.zeros(rx_voc_size)
            tmp_labels[list(map(lambda x: self.tokenizer.med_voc.word2idx[x], tokens))] = 1
            tmp_labels_multi = np.ones(rx_voc_size) * (-1)
            tmp_labels_multi[:len(tokens)] = list(map(lambda x: self.tokenizer.med_voc.word2idx[x], tokens))
            output_rx_labels.append(tmp_labels)
            output_dx_labels.append(tmp_labels_multi)

        cur_tensors = (torch.tensor(np.array(input_ids)).view(-1, self.seq_len),
                       torch.tensor(np.array(output_dx_labels), dtype=torch.float),
                       torch.tensor(np.array(output_rx_labels), dtype=torch.float))

        return cur_tensors



####################################
'''Prompt Dataset'''
####################################

class PromptEHRDataset(EHRDataset):

    def __init__(self, data_pd, tokenizer: EHRTokenizer, max_seq_len, prompt_num):
        
        super().__init__(data_pd, tokenizer, max_seq_len)
        self.tokenizer.vocab.add_sentence(['[PD]', '[PR]'])
        self.p_len = prompt_num + 1
        print('')


    def __len__(self):

        return len(self.records)

    
    def __getitem__(self, item):
        '''Only fill each record to max_len-1, and the remained one token is for prompt'''
        cur_id = self.sample_counter
        self.sample_counter += 1
        adm = copy.deepcopy(self.records[item])

        #p_len = 2 + 1   # the length of prompt, which should be freed for prompt
        p_len = self.p_len

        # cut the too long records 
        if len(adm[0]) >= self.seq_len - p_len:
            adm[0] = adm[0][:self.seq_len-p_len]
        if len(adm[1]) >= self.seq_len - p_len:
            adm[1] = adm[1][:self.seq_len-p_len]

        def fill_to_max(l, seq):
            while len(l) < seq:
                l.append('[PAD]')
            return l

        """extract input and output tokens
        """
        input_tokens = []  # (2*max_len*adm)
        output_dx_tokens = []  # (adm-1, l)
        output_rx_tokens = []  # (adm-1, l)

        input_tokens.extend(['[CLS]'] + fill_to_max(list(adm[0]), self.seq_len - p_len))    # shape: (2, max_len-1)
        input_tokens.extend(['[CLS]'] + fill_to_max(list(adm[1]), self.seq_len - p_len))
        output_rx_tokens.append(list(adm[2]))

        diag_prompt = np.zeros((len(input_tokens), 1))
        treat_prompt = np.ones((len(input_tokens), 1))
        prompt_tokens = np.concatenate([diag_prompt, treat_prompt], axis=0)

        """convert tokens to id
        """
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
        output_dx_labels = []  # (adm-1, dx_voc_size)
        output_rx_labels = []  # (adm-1, rx_voc_size)

        rx_voc_size = len(self.tokenizer.med_voc.word2idx)

        for tokens in output_rx_tokens:
            tmp_labels = np.zeros(rx_voc_size)
            tmp_labels[list(map(lambda x: self.tokenizer.med_voc.word2idx[x], tokens))] = 1
            tmp_labels_multi = np.ones(rx_voc_size) * (-1)
            tmp_labels_multi[:len(tokens)] = list(map(lambda x: self.tokenizer.med_voc.word2idx[x], tokens))
            output_rx_labels.append(tmp_labels)
            output_dx_labels.append(tmp_labels_multi)

        cur_tensors = (torch.tensor(np.array(input_ids)).view(-1, self.seq_len-(p_len-1)),
                       torch.tensor(np.array(output_dx_labels), dtype=torch.float),
                       torch.tensor(np.array(output_rx_labels), dtype=torch.float),
                       torch.tensor(prompt_tokens))

        return cur_tensors



