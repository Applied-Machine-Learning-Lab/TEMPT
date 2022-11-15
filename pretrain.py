import os
import argparse
import torch
from generators.pretrain_generator import PretrainGenerator
from generators.data import Voc
from trainers.pretrain_trainer import PretrainTrainer
from utils.utils import set_seed, log_res
from utils.logger import Logger

import setproctitle
setproctitle.setproctitle("MedRec-Pretrain")


parser = argparse.ArgumentParser()

# Required parameters
parser.add_argument("--model_name", 
                    default='TEMPT', 
                    type=str, 
                    required=False,
                    help="model name")
parser.add_argument("--dataset", 
                    default="eicu", 
                    choices=['eicu'], 
                    help="Choose the dataset")
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

# Other parameters
parser.add_argument("--attribute",
                    default=False,
                    action='store_true',
                    help="whether use attribute as one of self-supervised tasks")
parser.add_argument("--hospital",
                    default=False,
                    action="store_true",
                    help="whether train hopital-based prompt in pretrain stage")
parser.add_argument("--contrast",
                    default=False,
                    action="store_true",
                    help="whether use contrastive loss")
parser.add_argument("--tau",
                    default=0.1,
                    type=float,
                    help="the temperature factor")
parser.add_argument("--loss_weight",
                    default=1,
                    type=float,
                    help="the loss weight for contrastive loss and attribute prediction loss")
parser.add_argument("--graph",
                    default=False,
                    action='store_true',
                    help="if use ontology embedding")
parser.add_argument("--therhold",
                    default=0.2,
                    type=float,
                    help="therhold.")
parser.add_argument("--max_seq_length",
                    default=100,
                    type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. \n"
                            "Sequences longer than this will be truncated, and sequences shorter \n"
                            "than this will be padded.")
parser.add_argument("--train_batch_size",
                    default=64,
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
                    default=30.0,
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
                    help="random seed for initialization")
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

args.data_dir = './data/' + str(args.dataset) + '/handled/'
args.output_dir = os.path.join(args.pretrain_dir, str(args.seed))

set_seed(args.seed) # fix the random seed


def main():

    log_manager = Logger(args)  # initialize the log manager
    logger, writer = log_manager.get_logger()    # get the logger

    device = torch.device("cuda:"+str(args.gpu_id) if torch.cuda.is_available()
                          and not args.no_cuda else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)

    # generator is used to manage dataset
    generator = PretrainGenerator(args, logger, device)
    trainer = PretrainTrainer(args, logger, writer, device, generator)

    res = trainer.train()

    log_manager.end_log()   # delete the logger threads



if __name__ == "__main__":

    main()
