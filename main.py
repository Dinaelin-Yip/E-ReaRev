import argparse

from utils import create_logger
import torch
import numpy as np
import os
import time
#from Models.ReaRev.rearev import 
from train_model import Trainer_KBQA
from parsing import add_parse_args

os.chdir('/home/ye/ML/ReaRev_KGQA/')
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'
# export CUDA_VISIBLE_DEVICES=4

def seed_it(seed):
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    logger = create_logger(args)
    trainer = Trainer_KBQA(args=vars(args), model_name=args.model_name, logger=logger)
    
    '''
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    logger = create_logger(args)
    trainer = Trainer_KBQA(args=vars(args), model_name=args.model_name, logger=logger)
    '''

    if not args.is_eval:
        trainer.train(0, args.num_epoch - 1)
    else:
        assert args.load_experiment is not None
        if args.load_experiment is not None:
            ckpt_path = os.path.join(args.checkpoint_dir, args.load_experiment)
            # ckpt_path = os.path.join(args.checkpoint_dir, 'Mix.ckpt')
            print("Loading pre trained model from {}".format(ckpt_path))
        else:
            ckpt_path = None
        trainer.train_data = trainer.valid_data
        trainer.load_ckpt(ckpt_path)
        trainer.train(0, 0)
        # trainer.evaluate_single(ckpt_path)


def heyman(date):
    parser = argparse.ArgumentParser()
    parser = add_parse_args(parser)

    args = parser.parse_args()
    args.use_cuda = torch.cuda.is_available()

    seed_it(args.seed)
    
    if args.experiment_name == None:
        timestamp = str(int(time.time()))
        args.experiment_name = "{}-{}-{}".format(
            args.dataset,
            args.model_name,
            timestamp,
        )
    
    trainqa = 1
    completeness = 50
    
    args.edge_extension = True
    args.mean_extension = True
    
    # args.data_folder = f'new_data/meta/metaqa_{trainqa}_{completeness}/'
    args.data_folder = f'data/meta/metaqa-3hop-fake/'
    args.checkpoint_dir = f'checkpoint/{date}/meta_mix/{trainqa}_{completeness}/'
    args.num_iter = 2
    args.num_ins = 3
    args.num_gnn = 3
    args.fake_entities = False
    args.lr = 5e-4
    args.num_epoch = 75
    
    
    # args.data_folder = f'new_data/webqsp/webqsp_{trainqa}_{completeness}/'
    # args.checkpoint_dir = f'checkpoint/{date}/webqsp/{trainqa}_{completeness}/'
    # args.num_iter = 3
    # args.num_ins = 2
    # args.num_gnn = 2
    # args.fake_entities = True
    # args.lr = 3e-3
    # args.num_epoch = 100
    
    
    # args.data_folder = f'new_data/cwq/cwq_{trainqa}_{completeness}/'
    # args.checkpoint_dir = f'checkpoint/{date}/cwq/{trainqa}_{completeness}/'
    # args.num_iter = 2
    # args.num_ins = 3
    # args.num_gnn = 3
    # args.fake_entities = True
    # args.lr = 5e-4
    # args.num_epoch = 100
    
    main(args)
    
    
if __name__ == '__main__':
    for date in ['1213']:
        print('heyman')
        heyman(date)