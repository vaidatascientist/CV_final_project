# =============================================================================
import os
import numpy as np
import torch
import argparse
from model import SASNet
import warnings
import random
from datasets.loading_data import loading_data
import matplotlib.pyplot as plt
import csv

warnings.filterwarnings('ignore')

# define the GPU id to be used
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_args_parser():
    # define the argparse for the script
    parser = argparse.ArgumentParser('Inference setting', add_help=False)
    parser.add_argument('--model_path', type=str, help='path of pre-trained model')
    parser.add_argument('--data_path', type=str, help='root path of the dataset')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size in training')
    parser.add_argument('--log_para', type=int, default=1000, help='magnify the target density map')
    parser.add_argument('--block_size', type=int, default=32, help='patch size for feature level selection')

    return parser

# get the dataset
def prepare_dataset(args):
    return loading_data(args)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.cur_val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    # update the moving average
    def update(self, cur_val):
        self.cur_val = cur_val
        self.sum += cur_val
        self.count += 1
        self.avg = self.sum / self.count
    
def main(args):
    """the main process of inference"""

    with torch.no_grad():
        torch.cuda.empty_cache()

    test_loader, val_loader = prepare_dataset(args)   
    model = SASNet(args=args).cuda()
    model.load_state_dict(torch.load(args.model_path))
    i = 0
    print("start testing")
    with torch.no_grad():
        maes = AverageMeter()
        mses = AverageMeter()
        model.eval()
        for vi, data in enumerate(test_loader):
            print("in progress")
            img, gt_map = data
            if torch.cuda.is_available():
                img = img.cuda()
                gt_map = gt_map.type(torch.FloatTensor).cuda()

            
            pred_map = model(img)
            pred_map = pred_map.cpu().detach().numpy()
            gt_map = gt_map.cpu().detach().numpy()
            # evaluation over the batch
            for i_img in range(pred_map.shape[0]):
                pred_cnt = np.sum(pred_map[i_img], (1, 2)) / args.log_para
                gt_count = np.sum(gt_map[i_img], (1, 2))
                mae = abs(gt_count - pred_cnt)
                mse = (gt_count - pred_cnt) * (gt_count - pred_cnt)
                maes.update(mae)
                mses.update(mse)
            i = i + 1
        mae = maes.avg
        mse = np.sqrt(mses.avg) 
        
        # print the results
        print('=' * 50)
        print('    ' + '-' * 20)
        print('    [mae %.3f mse %.3f]' % (mae, mse))
        print('    ' + '-' * 20)
        print('=' * 50)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('SASNet inference', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
