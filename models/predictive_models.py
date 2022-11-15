from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import json
import math
import logging
import numpy as np

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.nn import LayerNorm
import torch.nn.functional as F
from utils.config import BertConfig
from models.bert_models import BERT, PreTrainedBertModel, BertLMPredictionHead, PromptBERT, TransformerBlock, gelu


logger = logging.getLogger(__name__)

CONFIG_NAME = 'bert_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'


def freeze_afterwards(model):
    for p in model.parameters():
        p.requires_grad = False


class TSNE(PreTrainedBertModel):
    def __init__(self, config: BertConfig, dx_voc=None, rx_voc=None):
        super(TSNE, self).__init__(config)

        self.bert = BERT(config, dx_voc, rx_voc)
        self.dx_voc = dx_voc
        self.rx_voc = rx_voc

        freeze_afterwards(self)

    def forward(self, output_dir, output_file='graph_embedding.tsv'):
        # dx_graph_emb = self.bert.embedding.ontology_embedding.dx_embedding.embedding
        # rx_graph_emb = self.bert.embedding.ontology_embedding.rx_embedding.embedding

        if not self.config.graph:
            print('save embedding not graph')
            rx_graph_emb = self.bert.embedding.word_embeddings(
                torch.arange(3, len(self.rx_voc.word2idx) + 3, dtype=torch.long))
            dx_graph_emb = self.bert.embedding.word_embeddings(
                torch.arange(len(self.rx_voc.word2idx) + 3, len(self.rx_voc.word2idx) + 3 + len(self.dx_voc.word2idx),
                             dtype=torch.long))
        else:
            print('save embedding graph')

            dx_graph_emb = self.bert.embedding.ontology_embedding.dx_embedding.get_all_graph_emb()
            rx_graph_emb = self.bert.embedding.ontology_embedding.rx_embedding.get_all_graph_emb()

        np.savetxt(os.path.join(output_dir, 'dx-' + output_file),
                   dx_graph_emb.detach().numpy(), delimiter='\t')
        np.savetxt(os.path.join(output_dir, 'rx-' + output_file),
                   rx_graph_emb.detach().numpy(), delimiter='\t')

        # def dump(prefix='dx-', emb):
        #     with open(prefix + output_file ,'w') as fout:
        #         m = emb.detach().cpu().numpy()
        #         for
        #         fout.write()


class ClsHead(nn.Module):
    def __init__(self, config: BertConfig, voc_size):
        super(ClsHead, self).__init__()
        self.cls = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), nn.ReLU(
        ), nn.Linear(config.hidden_size, voc_size))

    def forward(self, input):
        return self.cls(input)


class SelfSupervisedHead(nn.Module):
    def __init__(self, config: BertConfig, dx_voc_size, rx_voc_size):
        super(SelfSupervisedHead, self).__init__()
        self.multi_cls = nn.ModuleList([ClsHead(config, dx_voc_size), ClsHead(
            config, dx_voc_size), ClsHead(config, rx_voc_size), ClsHead(config, rx_voc_size)])

    def forward(self, dx_inputs, rx_inputs):
        # inputs (B, hidden)
        # output logits
        return self.multi_cls[0](dx_inputs), self.multi_cls[1](rx_inputs), self.multi_cls[2](dx_inputs), self.multi_cls[3](rx_inputs)


class ClsHeadHos(nn.Module):
    def __init__(self, config: BertConfig, voc_size):
        super(ClsHeadHos, self).__init__()
        self.cls = nn.Sequential(nn.Linear(config.hidden_size+20, config.hidden_size), nn.ReLU(
        ), nn.Linear(config.hidden_size, voc_size))

    def forward(self, input):
        return self.cls(input)


class SelfSupervisedHeadHos(nn.Module):
    def __init__(self, config: BertConfig, dx_voc_size, rx_voc_size):
        super(SelfSupervisedHeadHos, self).__init__()
        self.multi_cls = nn.ModuleList([ClsHeadHos(config, dx_voc_size), ClsHeadHos(
            config, dx_voc_size), ClsHeadHos(config, rx_voc_size), ClsHeadHos(config, rx_voc_size)])

    def forward(self, dx_inputs, rx_inputs):
        # inputs (B, hidden)
        # output logits
        return self.multi_cls[0](dx_inputs), self.multi_cls[1](rx_inputs), self.multi_cls[2](dx_inputs), self.multi_cls[3](rx_inputs)



class MappingHead(nn.Module):
    def __init__(self, config: BertConfig):
        super(MappingHead, self).__init__()
        self.dense = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                   nn.ReLU())

    def forward(self, input):
        return self.dense(input)



