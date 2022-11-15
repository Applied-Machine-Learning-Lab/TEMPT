import torch
import torch.nn as nn
import torch.nn.functional as F
from models.predictive_models import PreTrainedBertModel, MappingHead
from models.bert_models import BERT, PromptBERT, BertEmbeddings, PromptEmbeddings, TransformerBlock
from utils.config import BertConfig


class PMRec_Prompt(PreTrainedBertModel):
    def __init__(self, config: BertConfig, tokenizer, prompt_num):
        
        super().__init__(config)
        self.prompt_num = prompt_num
        self.bert = MultiPromptBERT(config, tokenizer.diag_voc, tokenizer.pro_voc, self.prompt_num)
        self.dense = nn.ModuleList([MappingHead(config), MappingHead(config)])
        self.cls = nn.Sequential(nn.Linear(2*config.hidden_size, 2*config.hidden_size),
                                nn.ReLU(), nn.Linear(2*config.hidden_size, len(tokenizer.med_voc.word2idx)))

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, dx_labels=None, rx_labels=None, epoch=None, prompt=None):
        """
        :param input_ids: [2, max_seq_len] (old: [2, adm_num, max_seq_len])
        :param rx_labels: [adm-1, rx_size]
        :param dx_labels: [adm-1, dx_size]
        :return:
        """
        token_types_ids = torch.cat([torch.zeros((input_ids.size(0), 1, input_ids.size(2))), torch.ones(
            (input_ids.size(0), 1, input_ids.size(2)))], dim=1).long().to(input_ids.device)
        token_types_ids = token_types_ids.repeat(1, 
            1 if input_ids.size(1)//2 == 0 else input_ids.size(1)//2, 1)    # (bs, 2, max_seq_len)
        # bert_pool: (bs, 2, H)
        _, dx_bert_pool = self.bert(input_ids[:, 0, :], token_types_ids[:, 0, :], input_prompt=prompt[:, 0, :])    # (bs, hidden_size)
        _, rx_bert_pool = self.bert(input_ids[:, 1, :], token_types_ids[:, 1, :], input_prompt=prompt[:, 1, :])    # (bs, hidden_size)
        loss = 0

        concat_vector = torch.cat([dx_bert_pool, rx_bert_pool], dim=-1) # (bs, 2*hidden_size)
        rx_logits = self.cls(concat_vector) # (bs, med_num)

        #rx_logits = torch.cat(rx_logits, dim=0)
        loss = F.binary_cross_entropy_with_logits(rx_logits, rx_labels)
        #    0.05 * F.multilabel_margin_loss(F.sigmoid(rx_logits), dx_labels.long())
        return loss, rx_logits



class MultiPromptBERT(PreTrainedBertModel):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, config: BertConfig, dx_voc=None, rx_voc=None, prompt_num=0):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__(config)

        self.prompt_num = prompt_num
        if config.graph:
            assert dx_voc is not None
            assert rx_voc is not None

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BertEmbeddings(config)
        
        # embedding for prompt
        self.prompt_embedding_list = nn.ModuleList([PromptEmbeddings(config) for _ in range(self.prompt_num)])

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_hidden_layers)])

        # pool first output
        # self.pooler = BertPooler(config)

        self.apply(self.init_bert_weights)

    def forward(self, x, token_type_ids=None, input_positions=None, input_sides=None, input_prompt=None):
        # attention masking for padded token
        mask = (torch.cat([x[:, 0].unsqueeze(1), self.prompt_num  * torch.ones((x.shape[0], self.prompt_num), device=x.device), x[:, 1:]], dim=1) > 1).unsqueeze(1).repeat(1, x.size(1)+self.prompt_num, 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, token_type_ids)   # (bs, max_seq_len, hidden_size)

        # get prompt and concat it to original input
        prompt = []
        for prompt_embedding in self.prompt_embedding_list:
            prompt.append(prompt_embedding(input_prompt.long()))
        prompt = torch.cat(prompt, dim=1)   # (bs, 5, hidden_size)
        x = torch.cat([x[:, 0, :].unsqueeze(1), prompt, x[:, 1:, :]], dim=1) # [CLS] should be put before the prompt

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x, x[:, 0]   # like rnn, only return the last hidden state




