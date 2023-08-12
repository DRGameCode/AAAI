import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default = 'Game', type = str,
                        help = 'Dataset to use')
    parser.add_argument('--seed', default = 2023, type = int,
                        help = 'seed for experiment')
    parser.add_argument('--embed_size', default = 32, type = int,
                        help = 'embedding size for all layer')
    parser.add_argument('--lr', default = 0.001, type = float,
                        help = 'learning rate')
    parser.add_argument('--weight_decay', default = 8e-8, type = float,
                        help = "weight decay for adam optimizer")
    parser.add_argument('--model', default = 'rec', type = str,
                        help = 'model selection')
    parser.add_argument('--epoch', default = 6000, type = int,
                        help = 'epoch number')
    parser.add_argument('--patience', default = 10, type = int,
                        help = 'early_stop validation')
    parser.add_argument('--batch_size', default = 2048, type = int,
                        help = 'batch size')
    parser.add_argument('--layers', default = 1, type = int,
                        help = 'layer number')
    parser.add_argument('--gpu', default = 0, type = int,
                        help = '-1 for cpu, 0 for gpu:0')
    parser.add_argument('--k_list', default = [50, 100, 150, 200, 250, 300], type = list,
                        help = 'topk evaluation')
    parser.add_argument('--p1', default = 0.02, type = float,
                        help = 'The ratio of redundant nodes to remove')
    parser.add_argument('--p2', default = 0.7, type = float,
                        help = 'The ratio of redundant nodes to remove')    
    parser.add_argument('--neg_number', default = 4, type = int,
                        help = 'negative sampler number for each positive pair')
    parser.add_argument('--metrics', default = ['recall', 'hit_ratio', 'coverage', 'precision'])

##########################################################################################

    parser.add_argument('--category_balance', default = 1, type = int,
                        help = 'whether make loss category balance')
    parser.add_argument('--beta_class', default = 0.9, type = float,
                        help = 'class re-balanced loss beta')
    
##########################################################################################

    parser.add_argument("--dgi-dropout", type=float, default=0.0, 
                        help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=512, 
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=3, 
                        help="number of hidden gcn layers")
    parser.add_argument("--dgi-lr", type=float, default=1e-3, 
                        help="dgi learning rate")
    parser.add_argument("--dgi-weight-decay", type=float, default=0.0, 
                        help="Weight for L2 loss")
    parser.add_argument("--n-dgi-epochs",type=int,default=300,
                        help="number of training epochs")
    parser.add_argument("--dgi-patience", type=int, default=30, 
                        help="early stop patience condition")
##########################################################################################

    parser.add_argument('--log', default = '', type = str,
                        help = 'log comments')

    args = parser.parse_args()
    return args

