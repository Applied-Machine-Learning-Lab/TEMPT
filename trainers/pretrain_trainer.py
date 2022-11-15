import os
import torch
import numpy as np
from tqdm import tqdm
from trainers.trainer import Trainer
from models.pretrain_models import PMRec_Pretrain_Contrastive
from utils.config import BertConfig
from utils.utils import t2n, metric_report, get_n_params
import time


class PretrainTrainer(Trainer):

    def __init__(self, args, logger, writer, device, generator):
        
        super().__init__(args, logger, writer, device, generator)


    def _create_model(self):
        '''Load pretrain model or not'''
        config = BertConfig(
            vocab_size_or_config_json_file=len(self.tokenizer.vocab.word2idx))
        config.graph = self.args.graph

        self.model = PMRec_Pretrain_Contrastive(config, self.tokenizer.diag_voc, self.tokenizer.pro_voc, self.args.tau, self.args.loss_weight)

        
        self.logger.info('# of model parameters: ' + str(get_n_params(self.model)))

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
            if self.args.attribute:
                input_ids, dx_labels, rx_labels, attri = batch # (bs, 2, max_seq_len), (bs, 1, med_num)
                loss, dx2dx, rx2dx, dx2rx, rx2rx = self.model(input_ids, dx_labels, rx_labels, attri)
            elif (self.args.hospital) & (not self.args.contrast):
                input_ids, dx_labels, rx_labels, hos_inputs = batch # (bs, 2, max_seq_len), (bs, 1, med_num)
                loss, dx2dx, rx2dx, dx2rx, rx2rx = self.model(input_ids, dx_labels, rx_labels, hos_inputs)
            elif (self.args.contrast) & (not self.args.hospital):
                input_ids, dx_labels, rx_labels, input_ids_raw = batch # (bs, 2, max_seq_len), (bs, 1, med_num)
                loss, dx2dx, rx2dx, dx2rx, rx2rx = self.model(input_ids, dx_labels, rx_labels, input_ids_raw)
            elif (self.args.contrast) & (self.args.hospital):
                input_ids, dx_labels, rx_labels, input_ids_raw, hos_inputs = batch # (bs, 2, max_seq_len), (bs, 1, med_num)
                loss, dx2dx, rx2dx, dx2rx, rx2rx = self.model(input_ids, dx_labels, rx_labels, input_ids_raw, hos_inputs)
            else:
                input_ids, dx_labels, rx_labels = batch # (bs, 2, max_seq_len), (bs, 1, med_num)
                loss, dx2dx, rx2dx, dx2rx, rx2rx = self.model(input_ids, dx_labels, rx_labels)
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

        dx2dx_y_preds = []
        rx2dx_y_preds = []
        dx_y_trues = []

        dx2rx_y_preds = []
        rx2rx_y_preds = []
        rx_y_trues = []
        for eval_input in tqdm(test_loader, desc=desc):
            eval_input = tuple(t.to(self.device) for t in eval_input)
            if self.args.attribute:
                input_ids, dx_labels, rx_labels, _ = eval_input
            elif (self.args.hospital) & (not self.args.contrast):
                input_ids, dx_labels, rx_labels, hos_inputs = eval_input
            elif (self.args.contrast) & (not self.args.hospital):
                input_ids, dx_labels, rx_labels, input_ids_raw = eval_input
            elif (self.args.contrast) & (self.args.hospital):
                input_ids, dx_labels, rx_labels, input_ids_raw, hos_inputs = eval_input
            else:
                input_ids, dx_labels, rx_labels = eval_input

            with torch.no_grad():
                if (self.args.hospital) & (not self.args.contrast):
                    dx2dx, rx2dx, dx2rx, rx2rx = self.model(input_ids, hos_inputs=hos_inputs)
                elif (self.args.contrast) & (not self.args.hospital):
                    dx2dx, rx2dx, dx2rx, rx2rx = self.model(input_ids, inputs_raw=input_ids_raw)
                elif (self.args.contrast) & (self.args.hospital):
                    dx2dx, rx2dx, dx2rx, rx2rx = self.model(input_ids, inputs_raw=input_ids_raw, hos_inputs=hos_inputs)
                else:
                    dx2dx, rx2dx, dx2rx, rx2rx = self.model(input_ids)
                dx2dx_y_preds.append(t2n(dx2dx))
                rx2dx_y_preds.append(t2n(rx2dx))
                dx2rx_y_preds.append(t2n(dx2rx))
                rx2rx_y_preds.append(t2n(rx2rx))

                dx_y_trues.append(t2n(dx_labels))
                rx_y_trues.append(t2n(rx_labels))

        self.logger.info('')
        #self.logger.info('dx2dx')
        #dx2dx_acc_container = metric_report(self.logger,
        #                                    np.concatenate(dx2dx_y_preds, axis=0), 
        #                                    np.concatenate(dx_y_trues, axis=0), 
        #                                    self.args.therhold)
        #self.logger.info('rx2dx')
        #rx2dx_acc_container = metric_report(self.logger,
        #                                    np.concatenate(rx2dx_y_preds, axis=0), 
        #                                    np.concatenate(dx_y_trues, axis=0), 
        #                                    self.args.therhold)
        #self.logger.info('dx2rx')
        #dx2rx_acc_container = metric_report(self.logger,
        #                                    np.concatenate(dx2rx_y_preds, axis=0), 
        #                                    np.concatenate(rx_y_trues, axis=0), 
        #                                    self.args.therhold)
        self.logger.info('rx2rx')
        rx2rx_acc_container = metric_report(self.logger,
                                            np.concatenate(rx2rx_y_preds, axis=0), 
                                            np.concatenate(rx_y_trues, axis=0), 
                                            self.args.therhold)

        #keep in history
        #for k, v in dx2dx_acc_container.items():
        #    self.writer.add_scalar(
        #        'eval_dx2dx/{}'.format(k), v, epoch)
        #for k, v in rx2dx_acc_container.items():
        #    self.writer.add_scalar(
        #        'eval_rx2dx/{}'.format(k), v, epoch)
        #for k, v in dx2rx_acc_container.items():
        #    self.writer.add_scalar(
        #        'eval_dx2rx/{}'.format(k), v, epoch)
        #for k, v in rx2rx_acc_container.items():
        #    self.writer.add_scalar(
        #        'eval_rx2rx/{}'.format(k), v, epoch)

        return rx2rx_acc_container



