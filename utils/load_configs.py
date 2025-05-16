import argparse
import sys
import torch


def get_link_prediction_args(is_evaluation: bool = False):
    """
    get the args for the link prediction task
    :param is_evaluation: boolean, whether in the evaluation process
    :return:
    """
    # arguments
    parser = argparse.ArgumentParser('Interface for the link prediction task')
    parser.add_argument('--dataset_name', type=str, help='dataset to be used', default='USLegis',
                        choices=['enron', 'USLegis', 'bitalpha', 'bitotc', 'Infectious', 'Haggle'])
    parser.add_argument('--batch_size', type=int, default=200, help='batch size')
    parser.add_argument('--prefix_name', type=str, default='preTraining', help='framework Name')
    parser.add_argument('--model_name', type=str, default='TBD', help='name of the model', choices=['TBD'])
    parser.add_argument('--gpu', type=int, default=0, help='number of gpu to use')
    parser.add_argument('--stride', type=int, default=2000, help='stride of two neighbor time blocks')
    parser.add_argument('--block_size', type=int, default=5000, help='size of time blocks')
    parser.add_argument('--hidden_dim', type=int, default=256, help='dim of hidden layers')
    parser.add_argument('--smoothness_weight', type=float, default=0.2, help='weight of the energy function')
    parser.add_argument('--sample_neighbor_strategy', default='recent', choices=['uniform', 'recent', 'time_interval_aware'], help='how to sample historical neighbors')
    parser.add_argument('--time_scaling_factor', default=1e-6, type=float, help='the hyperparameter that controls the sampling preference with time interval, '
                        'a large time_scaling_factor tends to sample more on recent links, 0.0 corresponds to uniform sampling, '
                        'it works when sample_neighbor_strategy == time_interval_aware')
    parser.add_argument('--high_order', action='store_true', default=False, help='use high-order neighbors')
    parser.add_argument('--num_layers', type=int, default=2, help='number of model layers')
    parser.add_argument('--time_feat_dim', type=int, default=256, help='dimension of the time embedding')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--num_epochs', type=int, default=150, help='number of epochs')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam', 'RMSprop'], help='name of optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--patience', type=int, default=20, help='patience for early stopping')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='ratio of validation set')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='ratio of test set')
    parser.add_argument('--num_runs', type=int, default=1, help='number of runs')
    parser.add_argument('--start_s', type=int, default=0, help='number of runs')
    parser.add_argument('--test_interval_epochs', type=int, default=10, help='how many epochs to perform testing once')
    parser.add_argument('--negative_sample_strategy', type=str, default='random', choices=['random', 'historical', 'inductive'],
                        help='strategy for the negative edge sampling')
    parser.add_argument('--load_best_configs', action='store_true', default=False, help='whether to load the best configurations')

    try:
        args = parser.parse_args()
        args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
        # args.device =  'cpu'
    except:
        parser.print_help()
        sys.exit()

    if args.load_best_configs:
        load_link_prediction_best_configs(args=args)

    return args


def load_link_prediction_best_configs(args: argparse.Namespace):
    """
    load the best configurations for the link prediction task
    :param args: argparse.Namespace
    :return:
    """
    # model specific settings
    if args.model_name in ['TBD']:
        if args.dataset_name in ['USLegis']:
            args.stride = 3000
            args.block_size = 5001
            args.dropout = 0.3
            args.hidden_dim = 128
            args.num_layers = 3
            args.smoothness_weight = 0.2
        elif args.dataset_name in ['enron']:
            args.stride = 2000
            args.block_size = 2222
            args.dropout = 0.4
            args.num_layers = 3
            args.smoothness_weight = 0.1
        elif args.dataset_name in ['bitalpha']:
            args.stride = 1000
            args.block_size = 1111
            args.dropout = 0.6
            args.num_layers = 3
            args.patience = 50
            args.smoothness_weight = 0.1
        elif args.dataset_name in ['bitotc']:
            args.stride = 1000
            args.block_size = 1053
            args.dropout = 0.6
            args.num_layers = 3
            args.smoothness_weight = 0.1
        elif args.dataset_name in ['Infectious']:
            args.stride = 1000
            args.block_size = 1176
            args.dropout = 0.6
            args.num_layers = 2
            args.patience = 50
            args.learning_rate = 5e-4
            args.smoothness_weight = 0.1
        elif args.dataset_name in ['Haggle']:
            args.stride = 1000
            args.block_size = 1111
            args.dropout = 0.2
            args.num_layers = 2
            args.learning_rate = 5e-4
            args.smoothness_weight = 0.2
    else:
        raise ValueError(f"Wrong value for model_name {args.model_name}!")
