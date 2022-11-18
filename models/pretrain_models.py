import torch
import torch.nn as nn
import torch.nn.functional as F
from models.bert_models import PreTrainedBertModel, BERT
from models.predictive_models import SelfSupervisedHead, SelfSupervisedHeadHos
from models.prompt_models import MultiHosPromptBERT
from utils.config import BertConfig


class PMRec_Pretrain(PreTrainedBertModel):
    def __init__(self, config: BertConfig, dx_voc=None, rx_voc=None):
        super(PMRec_Pretrain, self).__init__(config)
        self.dx_voc_size = len(dx_voc.word2idx)
        self.rx_voc_size = len(rx_voc.word2idx)

        self.bert = BERT(config, dx_voc, rx_voc)
        self.cls = SelfSupervisedHead(
            config, self.dx_voc_size, self.rx_voc_size)

        self.apply(self.init_bert_weights)

    def forward(self, inputs, dx_labels=None, rx_labels=None):
        # inputs (B, 2, max_len)
        # bert_pool (B, hidden)
        _, dx_bert_pool = self.bert(inputs[:, 0, :], torch.zeros(
            (inputs.size(0), inputs.size(2))).long().to(inputs.device))
        _, rx_bert_pool = self.bert(inputs[:, 1, :], torch.zeros(
            (inputs.size(0), inputs.size(2))).long().to(inputs.device))

        dx2dx, rx2dx, dx2rx, rx2rx = self.cls(dx_bert_pool, rx_bert_pool)
        # output logits
        if rx_labels is None or dx_labels is None:
            return F.sigmoid(dx2dx), F.sigmoid(rx2dx), F.sigmoid(dx2rx), F.sigmoid(rx2rx)
        else:
            loss = F.binary_cross_entropy_with_logits(dx2dx, dx_labels) + F.binary_cross_entropy_with_logits(rx2rx, rx_labels)
            return loss, F.sigmoid(dx2dx), F.sigmoid(rx2dx), F.sigmoid(dx2rx), F.sigmoid(rx2rx)



class PMRec_Pretrain_Contrastive(PreTrainedBertModel):
    def __init__(self, config: BertConfig, dx_voc=None, rx_voc=None, tau=0.1, loss_weight=1):
        super(PMRec_Pretrain_Contrastive, self).__init__(config)

        self.tau = tau
        self.loss_weight = loss_weight
        self.dx_voc_size = len(dx_voc.word2idx)
        self.rx_voc_size = len(rx_voc.word2idx)

        self.bert = BERT(config, dx_voc, rx_voc)
        self.cls = SelfSupervisedHead(
            config, self.dx_voc_size, self.rx_voc_size)
        self.project_dx = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        self.project_rx = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )

        self.apply(self.init_bert_weights)

    def forward(self, inputs, dx_labels=None, rx_labels=None, inputs_raw=None):
        # inputs (B, 2, max_len)
        # bert_pool (B, hidden)
        _, dx_bert_pool = self.bert(inputs[:, 0, :], torch.zeros(
            (inputs.size(0), inputs.size(2))).long().to(inputs.device))
        _, rx_bert_pool = self.bert(inputs[:, 1, :], torch.zeros(
            (inputs.size(0), inputs.size(2))).long().to(inputs.device))
        _, dx_bert_pool_raw = self.bert(inputs_raw[:, 0, :], torch.zeros(
            (inputs_raw.size(0), inputs_raw.size(2))).long().to(inputs_raw.device))
        _, rx_bert_pool_raw = self.bert(inputs_raw[:, 1, :], torch.zeros(
            (inputs_raw.size(0), inputs_raw.size(2))).long().to(inputs_raw.device))

        dx_bert_pool_raw = self.project_dx(dx_bert_pool_raw)
        rx_bert_pool_raw = self.project_rx(rx_bert_pool_raw)

        contrastive_loss = Contrastive_Loss(dx_bert_pool_raw, rx_bert_pool_raw, tau=self.tau) + \
                           Contrastive_Loss(rx_bert_pool_raw, dx_bert_pool_raw, tau=self.tau)

        dx2dx, rx2dx, dx2rx, rx2rx = self.cls(dx_bert_pool, rx_bert_pool)
        # output logits
        if rx_labels is None or dx_labels is None:
            return F.sigmoid(dx2dx), F.sigmoid(rx2dx), F.sigmoid(dx2rx), F.sigmoid(rx2rx)
        else:
            #loss = F.binary_cross_entropy_with_logits(dx2dx, dx_labels) + F.binary_cross_entropy_with_logits(rx2rx, rx_labels) + self.loss_weight * contrastive_loss
            loss = contrastive_loss + self.loss_weight * contrastive_loss
            return loss, F.sigmoid(dx2dx), F.sigmoid(rx2dx), F.sigmoid(dx2rx), F.sigmoid(rx2rx)




def Contrastive_Loss(X, Y, tau):
    '''
    X: (bs, hidden_size), Y: (bs, hidden_size)
    tau: the temperature factor
    '''
    #sim_matrix = X.mm(Y.t())    # (bs, bs)
    sim_matrix = F.cosine_similarity(X.unsqueeze(1), Y.unsqueeze(0), dim=2)
    pos = torch.exp(torch.diag(sim_matrix) / tau).unsqueeze(0)   # (1, bs)
    neg = torch.sum(torch.exp(sim_matrix / tau), dim=0) - pos     # (1, bs)
    loss = - torch.log(pos / neg)
    loss = torch.mean(loss)

    return loss
    




