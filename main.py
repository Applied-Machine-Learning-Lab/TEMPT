import os
import argparse

import numpy as np
import pandas as pd
import torch

from generators.data import Voc
from generators.finetune_generator import FinetuneGenerator
from generators.prompt_generator import PromptGenerator
from trainers.finetune_trainer import FinetuneTrainer
from trainers.prompt_trainer import PromptTrainer
from utils.utils import set_seed, log_res, log_res_multi, log_efficiency
from utils.logger import Logger

import setproctitle
setproctitle.setproctitle("MedRec-Finetune")


parser = argparse.ArgumentParser()

# Required parameters
parser.add_argument("--model_name", 
                    default='GBert-predict', 
                    type=str, 
                    required=False,
                    help="model name")
parser.add_argument("--dataset", 
                    default="eicu", 
                    choices=['eicu', 'mimic'], 
                    help="Choose the dataset")
parser.add_argument("--backbone",
                    default='bert',
                    choices=['bert', 'gamenet', 'leap', 'cognet'],
                    help='choose the general medication recommendation model')
parser.add_argument("--demo", 
                    default=False, 
                    action='store_true', 
                    help='whether run demo')
parser.add_argument("--pretrain_dir", 
                    default='./saved/eicu/default', 
                    type=str, 
                    required=False,
                    help="pretraining model")
parser.add_argument("--train_file", 
                    default='data-single-visit.pkl', 
                    type=str, 
                    required=False,
                    help="training data file.")
parser.add_argument("--output_dir",
                    default='./saved/',
                    type=str,
                    required=False,
                    help="The output directory where the model checkpoints will be written.")
parser.add_argument("--out_exp",
                    default='./log/result.json',
                    type=str,
                    help="The output json for multiple experiments of multiple centers")
parser.add_argument("--check_path",
                    default='',
                    type=str,
                    help="the save path of checkpoints for different running")

# Other parameters
parser.add_argument("--hospital",
                    default=False,
                    action="store_true",
                    help="whether train hopital-based prompt in pretrain stage")
parser.add_argument("--use_pretrain",
                    default=False,
                    action='store_true',
                    help="is use pretrain")
parser.add_argument("--use_prompt",
                    default=False,
                    action='store_true',
                    help='whether use prompt')
parser.add_argument("--att_prompt",
                    default=False,
                    action="store_true",
                    help="whether use the attention prompt")
parser.add_argument("--prompt_num",
                    default=2,
                    type=int,
                    help="the number of prompt")
parser.add_argument("--freeze", 
                    default=False,
                    action="store_true",
                    help="Whether freeze some layers of the model for finetuning")
parser.add_argument("--graph",
                    default=False,
                    action='store_true',
                    help="if use ontology embedding")
parser.add_argument("--therhold",
                    default=0.3,
                    type=float,
                    help="therhold.")
parser.add_argument("--max_seq_length",
                    default=100,
                    type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. \n"
                            "Sequences longer than this will be truncated, and sequences shorter \n"
                            "than this will be padded.")
parser.add_argument("--do_train",
                    default=False,
                    action='store_true',
                    help="Whether to run training.")
parser.add_argument("--do_eval",
                    default=True,
                    action='store_true',
                    help="Whether to run on the dev set.")
parser.add_argument("--do_test",
                    default=True,
                    action='store_true',
                    help="Whether to run on the test set.")
parser.add_argument("--single_train",
                    default=False,
                    action='store_true',
                    help='Train one single hospital or all of hospitals')
parser.add_argument("--single_test",
                    default=False,
                    action='store_true',
                    help='Test one single hospital or all of hospitals')
parser.add_argument("--hos_id",
                    default=79,
                    type=int,
                    help='Test the id-hospital')
parser.add_argument("--train_batch_size",
                    default=128,
                    type=int,
                    help="Total batch size for training.")
parser.add_argument("--learning_rate",
                    default=5e-4,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--l2",
                    default=0,
                    type=float,
                    help='The L2 regularization')
parser.add_argument("--num_train_epochs",
                    default=30,
                    type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--lr_dc_step",
                    default=1000,
                    type=int,
                    help='every n step, decrease the lr')
parser.add_argument("--lr_dc",
                    default=0,
                    type=float,
                    help='how many learning rate to decrease')
parser.add_argument("--patience",
                    type=int,
                    default=10,
                    help='How many steps to tolerate the performance decrease while training')
parser.add_argument("--no_cuda",
                    action='store_true',
                    help="Whether not to use CUDA when available")
parser.add_argument('--seed',
                    type=int,
                    default=42,
                    help="random seed for different data split")
parser.add_argument('--tseed',
                    default=42,
                    type=int,
                    help='random seed for training')
parser.add_argument("--warmup_proportion",
                    default=0.1,
                    type=float,
                    help="Proportion of training to perform linear learning rate warmup for. "
                            "E.g., 0.1 = 10%% of training.")
parser.add_argument('--gpu_id',
                    default=0,
                    type=int,
                    help='The device id.')
parser.add_argument('--num_workers',
                    default=0,
                    type=int,
                    help='The number of workers in dataloader')
parser.add_argument("--log", 
                    default=False,
                    action="store_true",
                    help="whether create a new log file")


args = parser.parse_args()
args.model_name = 'prompt'


args.model_name = args.model_name + '/' + args.check_path

if args.single_train:
    args.single_test = True

args.data_dir = './data/' + str(args.dataset) + '/handled/'
args.output_dir = args.output_dir + str(args.dataset) + '/'
args.output_dir = os.path.join(args.output_dir, args.model_name)
args.pretrain_dir = os.path.join(args.pretrain_dir, str(args.seed))

set_seed(args.seed) # fix the random seed


def main():

    log_manager = Logger(args)  # initialize the log manager
    logger, writer = log_manager.get_logger()    # get the logger

    device = torch.device("cuda:"+str(args.gpu_id) if torch.cuda.is_available()
                          and not args.no_cuda else "cpu")

    if not args.do_train and not args.do_eval:

        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    os.makedirs(args.output_dir, exist_ok=True)

    # generator is used to manage dataset
    generator = PromptGenerator(args, logger, device)
    trainer = PromptTrainer(args, logger, writer, device, generator)

    res, best_epoch, train_time = trainer.train()
    fp_num, ap_num = trainer.get_model_param_num()  # freeze parameter num, active parameter num

    log_manager.end_log()   # delete the logger threads

    if args.log:
        if (args.single_test) & (not args.single_train):
            log_res_multi(args, res, log_manager.get_now_str(), args.out_exp)
        else:
            log_res(args, res, log_manager.get_now_str(), args.out_exp)
            
        log_efficiency(best_epoch, train_time, fp_num, ap_num, args, log_manager.get_now_str())



if __name__ == "__main__":

    main()



