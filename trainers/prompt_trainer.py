import os
import torch
import numpy as np
from tqdm import tqdm
from trainers.trainer import Trainer
from models.prompt_models import PMRec_Prompt
from utils.config import BertConfig
from utils.utils import t2n, metric_report
import time


class PromptTrainer(Trainer):

    def __init__(self, args, logger, writer, device, generator):
        
        super().__init__(args, logger, writer, device, generator)

        if self.args.freeze:

            self._freeze()


    def _create_model(self):
        '''Load pretrain model or not'''
        if self.args.use_pretrain:
            self.logger.info("Use Pretraining model")
            self.model = PMRec_Prompt.from_pretrained(self.args.pretrain_dir, 
                                                        tokenizer=self.tokenizer, 
                                                        logger=self.logger,
                                                        prompt_num=self.args.prompt_num)
        else:
            config = BertConfig(vocab_size_or_config_json_file=len(self.tokenizer.vocab.word2idx))
            config.graph = self.args.graph
            self.model = PMRec_Prompt(config, self.tokenizer)
        
        self.model.to(self.device)


    def _train_one_epoch(self, epoch):
        
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        train_time = []

        self.model.train()
        prog_iter = tqdm(self.train_loader, leave=False, desc='Training')
        for batch in prog_iter:

            batch = tuple(t.to(self.device) for t in batch)

            train_start = time.time()
            input_ids, dx_labels, rx_labels, prompt = batch # (bs, 2, max_seq_len), (bs, 1, med_num)
            input_ids, dx_labels, rx_labels, prompt = input_ids.squeeze(
                dim=1), dx_labels.squeeze(dim=1), rx_labels.squeeze(dim=1), prompt.long()
            loss, rx_logits = self.model(input_ids, dx_labels=dx_labels, rx_labels=rx_labels, prompt=prompt,
                                    epoch=epoch)
            loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += 1
            nb_tr_steps += 1

            # Display loss
            prog_iter.set_postfix(loss='%.4f' % (tr_loss / nb_tr_steps))

            self.optimizer.step()
            self.optimizer.zero_grad()

            train_end = time.time()
            train_time.append(train_end-train_start)

        self.writer.add_scalar('train/loss', tr_loss / nb_tr_steps, epoch)

        return np.array(train_time).mean()


    def eval(self, epoch=0, test=False):

        print('')
        if test:
            self.logger.info("\n----------------------------------------------------------------")
            self.logger.info("********** Running test **********")
            self.logger.info("  Num examples = %d", self.generator.get_statistics()[2])
            desc = 'Testing'
            model_state_dict = torch.load(os.path.join(self.args.output_dir, 'pytorch_model.bin'))
            self.model.load_state_dict(model_state_dict)
            self.model.to(self.device)
            test_loader = self.test_loader
        
        else:
            self.logger.info("\n----------------------------------")
            self.logger.info("********** Epoch: %d eval **********" % epoch)
            desc = 'Evaluating'
            test_loader = self.eval_loader
        
        self.model.eval()

        rx_y_preds = []
        rx_y_trues = []
        for eval_input in tqdm(test_loader, desc=desc):
            eval_input = tuple(t.to(self.device) for t in eval_input)
            input_ids, dx_labels, rx_labels, prompt = eval_input
            input_ids, dx_labels, rx_labels, prompt = input_ids.squeeze(
            ), dx_labels.squeeze(), rx_labels.squeeze(dim=1), prompt.squeeze(dim=1)
            with torch.no_grad():
                loss, rx_logits = self.model(
                    input_ids, dx_labels=dx_labels, rx_labels=rx_labels, prompt=prompt)
                rx_y_preds.append(t2n(torch.sigmoid(rx_logits)))
                rx_y_trues.append(t2n(rx_labels))

        self.logger.info('')
        rx_acc_container = metric_report(self.logger, 
                                         np.concatenate(rx_y_preds, axis=0), 
                                         np.concatenate(rx_y_trues, axis=0),
                                         self.args.therhold)
        for k, v in rx_acc_container.items():
            self.writer.add_scalar('eval/{}'.format(k), v, epoch)

        return rx_acc_container

    
    def _freeze(self):
        # freeze all of bert parameters in the model except for prompt 
        for name, param in self.model.named_parameters():
    
            if 'bert' in name:
                if 'prompt' in name:
                    continue
                else:
                    param.requires_grad = False

        return 0






