import argparse
import os
import random

import numpy as np
import scipy.sparse as sp
from pprint import pprint
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data

import data_utils
from CDAE import CDAE
from LightGCN import LightGCN
from model import NCF, MF
from train import TrainModel
from utils import get_top_k
from configs import get_config

def main(param):
    cudnn.benchmark = True
    torch.manual_seed(param['seed'])  # cpu
    torch.cuda.manual_seed(param['seed'])  # gpu
    np.random.seed(param['seed'])  # numpy
    random.seed(param['seed'])  # random and transforms
    torch.backends.cudnn.enabled =  True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # cudnn

    def worker_init_fn(worker_id):
        np.random.seed(param['seed'] + worker_id)

    data_path = f"{param['datadir']}/{param['dataset']}/"

    ############################## PREPARE DATASET ##########################
    train_data, valid_data, test_data_pos, test_data_noisy, user_pos, user_num, item_num, train_mat, valid_data_pos, \
    clean_test_mat, noisy_test_mat, train_data_noisy, valid_data_noisy = data_utils.load_all(param['dataset'], data_path)

    # construct the train and test datasets
    train_dataset = data_utils.NCFData(
        train_data, item_num, train_mat, NSR=0, is_training=0, noisy_or_not=train_data_noisy)
    valid_dataset = data_utils.NCFData(
        valid_data, item_num, train_mat, NSR=param.get('NSR', 1), is_training=1, noisy_or_not=valid_data_noisy)

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=param.get('batch_size', 2048), shuffle=True, num_workers=0, pin_memory=True,
                                   worker_init_fn=worker_init_fn)
    valid_loader = data.DataLoader(valid_dataset,
                                   batch_size=param.get('batch_size', 2048), shuffle=True, num_workers=0, pin_memory=True,
                                   worker_init_fn=worker_init_fn)

    print("data loaded! user_num:{}, item_num:{} train_data_len:{} test_user_num:{}".format(user_num, item_num,
                                                                                            len(train_data),
                                                                                            len(test_data_pos)))
    ########################### CREATE MODEL #################################
    if param['model'] == 'NeuMF-pre':  # pre-training. Not used in our work.
        GMF_folder = param['folder'] +  'GMF.pth'
        MLP_folder = param['folder'] +  'MLP.pth'
        NeuMF_folder = param['folder'] +  'NeuMF.pth'
        assert os.path.exists(GMF_folder), 'lack of GMF model'
        assert os.path.exists(MLP_folder), 'lack of MLP model'
        GMF_model = torch.load(GMF_folder)
        MLP_model = torch.load(MLP_folder)
    else:
        GMF_model = None
        MLP_model = None

    if param['model'] == 'LightGCN':
        try:
            norm_adj = sp.load_npz(data_path + param['dataset'] + '_s_pre_adj_mat.npz')
            print("successfully loaded...")
        except:
            norm_adj = data_utils.create_adj_mat(train_mat, user_num, item_num, param['folder'] + f"/{param['dataset']}")
        model = LightGCN(user_num, item_num, norm_adj, device=param['device']).to(param['device'])

    elif param['model'] == 'CDAE':
        model = CDAE(user_num, item_num).to(param['device'])
    elif param['model'] == 'MF':
        model = MF(user_num, item_num, param.get("factor_num", 32)).to(param.get("device", 'cuda'))
    else:
        model = NCF(user_num, item_num, param.get("factor_num", 32), param.get('num_layers', 3), dropout=param.get('dropout', 0),
                    model=param['model'], GMF_model=GMF_model, MLP_model=MLP_model)

    if not os.path.exists(param['folder']):
        os.mkdir(param['folder'])

    train_loader.dataset.ng_sample()

    train_model = TrainModel(param, model, user_num, item_num)
    results = train_model.train(train_loader, valid_loader, test_data_pos,
                                valid_data_pos, test_data_noisy, train_mat, user_pos)
    clean_precision, clean_recall, clean_NDCG, clean_MRR = results
    return np.concatenate([clean_precision, clean_recall, clean_NDCG, clean_MRR])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # flexible parameters
    parser.add_argument('--datadir', type=str, default=r'./data')
    parser.add_argument('--folder', type=str, default='./output')
    parser.add_argument('--model', type=str, default='GMF')
    parser.add_argument('--dataset', type=str, default='ml-100k',
                        help='dataset used for training, options: amazon_book, yelp, adressa')
    parser.add_argument("--epochs", type=int, default=15, help="training epoches")
    parser.add_argument("--top_k", type=list, default=[3, 5, 10, 20], help="compute metrics@top_k")
    parser.add_argument("--method", type=str, default=None, help='DeCA, DeCAp or None')
    parser.add_argument("--C_1", default=1000, type=int, help='the large number used in DP')
    parser.add_argument("--C_2", default=10, type=int, help='the large number used in DN')
    parser.add_argument("--alpha", type=int, default=1, help='weight between two KL divergence')
    parser.add_argument("--lambda0", type=float, default=1.0, help='regularization parameter')
    parser.add_argument("--denoise_type", type=str, default='both', help='DP, DF or both')
    parser.add_argument("--iterative", default=True, type=bool, help='whether to use iterative training routine')
    parser.add_argument("--early_stop", type=int, default=False, help='whether to early stop according the validation loss')
    parser.add_argument("--save_model", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2020)

    # following parameters are fixed during my implementation
    parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=2048, help="batch size for training")
    parser.add_argument("--eval_freq", type=int, default=100, help="the freq of eval")
    parser.add_argument("--factor_num", type=int, default=32, help="predictive factors numbers in the model")
    parser.add_argument("--num_layers", type=int, default=3, help="number of layers in MLP model")
    parser.add_argument("--NSR", type=int, default=1, help="sample negative items for training")
    parser.add_argument("--device", default='cuda', help='cuda or cpu')
    parser.add_argument("--emb_dim", type=int, default=32, help='embedding dimension of the Gamma model')
    parser.add_argument("--early_stop_rounds", type=int, default=2,
                        help='early stop after how many iteration rounds non decreasing')

    parser.add_argument("--default", action='store_true', default=False)
    args = parser.parse_args()

    if args.default == True:
        param = get_config(model=args.model, dataset= args.dataset, method=args.method)
        param['method'] = args.method
        param['top_k'] = get_top_k(args.dataset)
        param['seed'] = args.seed
        param['denoise_type'] = args.denoise_type
        param['datadir'] = args.datadir
        param['folder'] = args.folder
        param['model'] = args.model
        param['dataset'] = args.dataset
        param['save_model'] = args.save_model
        param['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        main(param)
    else:
        main({
            'datadir': args.datadir,
            'folder': args.folder,
            'model': args.model,
            'dataset': args.dataset,
            'epochs': args.epochs,
            'top_k': args.top_k,
            'method': args.method,
            'C_2': args.C_2,
            'C_1': args.C_1,
            'alpha': args.alpha,
            'lambda0': args.lambda0,
            'denoise_type': args.denoise_type,
            'iterative': args.iterative,
            'early_stop': args.early_stop,
            'save_model': args.save_model,
            "seed": args.seed,
            'dropout': args.dropout,
            'lr': args.lr,
            'batch_size': args.batch_size,
            'eval_freq': args.eval_freq,
            'factor_num': args.factor_num,
            'num_layers': args.num_layers,
            'NSR': args.NSR,
            'device': args.device,
            'emb_dim': args.emb_dim,
            'early_stop_rounds': args.early_stop_rounds,
        })