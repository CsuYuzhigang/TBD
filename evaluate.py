import logging
import time
import sys
import os
from pathlib import Path

import numpy as np
import warnings
import json
import torch.nn as nn

from models.TBD import TBD
from models.modules import MergeLayer
from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes, get_historical_neighbor_sampler, \
    create_temporal_blocks
from utils.utils import get_neighbor_sampler, NegativeEdgeSampler
from utils.evaluate_models_utils import evaluate, evaluate_mixture
from utils.DataLoader import get_idx_data_loader, get_link_prediction_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_prediction_args

if __name__ == "__main__":

    warnings.filterwarnings('ignore')

    # get arguments
    args = get_link_prediction_args(is_evaluation=True)

    # get data for training, validation and testing
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = \
        get_link_prediction_data(dataset_name=args.dataset_name, val_ratio=args.val_ratio, test_ratio=args.test_ratio)
    node_raw_features = node_raw_features[1:, :]
    edge_raw_features = edge_raw_features[1:, :]

    # initialize validation and test neighbor sampler to retrieve temporal graph
    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                 time_scaling_factor=args.time_scaling_factor, seed=1)
    full_edge_neighbor_sampler = get_historical_neighbor_sampler(data=full_data,
                                                                 sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                                 time_scaling_factor=args.time_scaling_factor, seed=1)

    # initialize negative samplers, set seeds for validation and testing so negatives are the same across different runs
    # in the inductive setting, negatives are sampled only amongst other new nodes
    if args.negative_sample_strategy != 'random':
        val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids,
                                                   interact_times=full_data.node_interact_times, last_observed_time=train_data.node_interact_times[-1],
                                                   negative_sample_strategy=args.negative_sample_strategy, seed=0)
        new_node_val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_val_data.src_node_ids, dst_node_ids=new_node_val_data.dst_node_ids,
                                                            interact_times=new_node_val_data.node_interact_times, last_observed_time=train_data.node_interact_times[-1],
                                                            negative_sample_strategy=args.negative_sample_strategy, seed=1)
        test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids,
                                                    interact_times=full_data.node_interact_times, last_observed_time=val_data.node_interact_times[-1],
                                                    negative_sample_strategy=args.negative_sample_strategy, seed=2)
        new_node_test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_test_data.src_node_ids, dst_node_ids=new_node_test_data.dst_node_ids,
                                                             interact_times=new_node_test_data.node_interact_times, last_observed_time=val_data.node_interact_times[-1],
                                                             negative_sample_strategy=args.negative_sample_strategy, seed=3)
    else:
        val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=0)
        new_node_val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_val_data.src_node_ids, dst_node_ids=new_node_val_data.dst_node_ids, seed=1)
        test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=2)
        new_node_test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_test_data.src_node_ids, dst_node_ids=new_node_test_data.dst_node_ids, seed=3)

    # get data loaders
    val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    new_node_val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    new_node_test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)

    val_metric_all_runs, new_node_val_metric_all_runs, test_metric_all_runs, new_node_test_metric_all_runs = [], [], [], []

    blocks, start_times, end_times = create_temporal_blocks(dataset_name=args.dataset_name,
                                                            node_raw_features=node_raw_features,
                                                            edge_raw_features=edge_raw_features,
                                                            full_data=full_data,
                                                            stride=args.stride,
                                                            block_size=args.block_size,
                                                            device=args.device,
                                                            drop_last=False,
                                                            evaluate=True)
    evaluate_func = evaluate
    if args.dataset_name in ['USLegis', 'enron']:
        evaluate_func = evaluate_mixture  # mixture temporal blocks prediction

    for run in range(args.start_s, args.num_runs):

        set_random_seed(seed=run)

        args.seed = run
        args.load_model_name = f'{args.model_name}_seed{args.seed}'
        args.save_result_name = f'{args.negative_sample_strategy}_negative_sampling_{args.model_name}_seed{args.seed}'

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_result_name}/", exist_ok=True)
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_result_name}/{str(time.time())}.log")
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        # # add wandb logInfo
        # run_dir = Path(
        #     f"./results/")
        # if not run_dir.exists():
        #     os.makedirs(str(run_dir))

        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts. **********")

        logger.info(f'configuration is {args}')

        # create model
        if args.model_name in ['TBD']:
            dynamic_backbone = TBD(num_nodes=node_raw_features.shape[0],
                                   in_feats=node_raw_features.shape[1],
                                   time_dim=args.time_feat_dim,
                                   hidden_dim=args.hidden_dim,
                                   out_dim=node_raw_features.shape[1],
                                   start_times=start_times,
                                   end_times=end_times,
                                   device=args.device,
                                   num_gnn_layers=args.num_layers,
                                   dropout=args.dropout,
                                   use_batch=False,
                                   batch_size=args.batch_size)
        else:
            raise ValueError(f"Wrong value for model_name {args.model_name}!")

        link_predictor = MergeLayer(input_dim1=node_raw_features.shape[1], input_dim2=node_raw_features.shape[1],
                                    input_dim3=0,
                                    hidden_dim=node_raw_features.shape[1] // 2, output_dim=1)

        model = nn.Sequential(dynamic_backbone, link_predictor)
        logger.info(f'model -> {model}')
        logger.info(f'model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, '
                    f'{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.')

        # load the saved model
        load_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.prefix_name}_{args.load_model_name}/"
        early_stopping = EarlyStopping(patience=0, save_model_folder=load_model_folder,
                                       save_model_name=args.load_model_name, logger=logger, model_name=args.model_name)
        early_stopping.load_checkpoint(model, map_location='cpu')

        model = convert_to_gpu(model, device=args.device)
        # put the node raw messages of memory-based models on device

        loss_func = nn.BCELoss()
        block_reprs_list = None

        # evaluate the best model
        logger.info(f'get final performance on dataset {args.dataset_name}...')

        val_losses, val_metrics = evaluate_func(model=model,
                                           evaluate_idx_data_loader=val_idx_data_loader,
                                           evaluate_neg_edge_sampler=val_neg_edge_sampler,
                                           evaluate_data=val_data,
                                           blocks=blocks,
                                           end_times=end_times,
                                           smoothness_weight=args.smoothness_weight,
                                           device=args.device)

        new_node_val_losses, new_node_val_metrics = evaluate_func(model=model,
                                                             evaluate_idx_data_loader=new_node_val_idx_data_loader,
                                                             evaluate_neg_edge_sampler=new_node_val_neg_edge_sampler,
                                                             evaluate_data=new_node_val_data,
                                                             blocks=blocks,
                                                             end_times=end_times,
                                                             smoothness_weight=args.smoothness_weight,
                                                             device=args.device)

        test_losses, test_metrics = evaluate_func(model=model,
                                             evaluate_idx_data_loader=test_idx_data_loader,
                                             evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                             evaluate_data=test_data,
                                             blocks=blocks,
                                             end_times=end_times,
                                             smoothness_weight=args.smoothness_weight,
                                             device=args.device)

        new_node_test_losses, new_node_test_metrics = evaluate_func(model=model,
                                                               evaluate_idx_data_loader=new_node_test_idx_data_loader,
                                                               evaluate_neg_edge_sampler=new_node_test_neg_edge_sampler,
                                                               evaluate_data=new_node_test_data,
                                                               blocks=blocks,
                                                               end_times=end_times,
                                                               smoothness_weight=args.smoothness_weight,
                                                               device=args.device)

        # store the evaluation metrics at the current run
        val_metric_dict, new_node_val_metric_dict, test_metric_dict, new_node_test_metric_dict = {}, {}, {}, {}

        logger.info(f'validate loss: {np.mean(val_losses):.4f}')
        for metric_name in val_metrics[0].keys():
            average_val_metric = np.mean([val_metric[metric_name] for val_metric in val_metrics])
            logger.info(f'validate {metric_name}, {average_val_metric:.4f}')
            val_metric_dict[metric_name] = average_val_metric

        logger.info(f'new node validate loss: {np.mean(new_node_val_losses):.4f}')
        for metric_name in new_node_val_metrics[0].keys():
            average_new_node_val_metric = np.mean([new_node_val_metric[metric_name] for new_node_val_metric in new_node_val_metrics])
            logger.info(f'new node validate {metric_name}, {average_new_node_val_metric:.4f}')
            new_node_val_metric_dict[metric_name] = average_new_node_val_metric

        logger.info(f'test loss: {np.mean(test_losses):.4f}')
        for metric_name in test_metrics[0].keys():
            average_test_metric = np.mean([test_metric[metric_name] for test_metric in test_metrics])
            logger.info(f'test {metric_name}, {average_test_metric:.4f}')
            test_metric_dict[metric_name] = average_test_metric

        logger.info(f'new node test loss: {np.mean(new_node_test_losses):.4f}')
        for metric_name in new_node_test_metrics[0].keys():
            average_new_node_test_metric = np.mean([new_node_test_metric[metric_name] for new_node_test_metric in new_node_test_metrics])
            logger.info(f'new node test {metric_name}, {average_new_node_test_metric:.4f}')
            new_node_test_metric_dict[metric_name] = average_new_node_test_metric

        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

        val_metric_all_runs.append(val_metric_dict)
        new_node_val_metric_all_runs.append(new_node_val_metric_dict)
        test_metric_all_runs.append(test_metric_dict)
        new_node_test_metric_all_runs.append(new_node_test_metric_dict)

        # avoid the overlap of logs
        if run < args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        # save model result
        result_json = {
            "validate metrics": {metric_name: f'{val_metric_dict[metric_name]:.4f}' for metric_name in val_metric_dict},
            "new node validate metrics": {metric_name: f'{new_node_val_metric_dict[metric_name]:.4f}' for metric_name in new_node_val_metric_dict},
            "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict},
            "new node test metrics": {metric_name: f'{new_node_test_metric_dict[metric_name]:.4f}' for metric_name in new_node_test_metric_dict}
        }

        result_json = json.dumps(result_json, indent=4)

        save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}"
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(save_result_folder, f"{args.save_result_name}.json")
        with open(save_result_path, 'w') as file:
            file.write(result_json)
        logger.info(f'save negative sampling results at {save_result_path}')

        # config_saver.end()
    # store the average metrics at the log of the last run
    logger.info(f'metrics over {args.num_runs} runs:')

    # # add wandb logInfo
    # run_dir = Path(
    #     f"./results/")
    # if not run_dir.exists():
    #     os.makedirs(str(run_dir))
    args.seed = -1

    average_val_metric, new_node_val_metric = {}, {}
    for metric_name in val_metric_all_runs[0].keys():
        list_mets = [val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]
        average_val_metric[metric_name] = np.mean(list_mets)
        logger.info(
            f'validate {metric_name}, {list_mets}')
        logger.info(
            f'average validate {metric_name}, {np.mean(list_mets):.4f} '
            f'± {np.std(list_mets, ddof=1):.4f}')

    for metric_name in new_node_val_metric_all_runs[0].keys():
        list_mets = [new_node_val_metric_single_run[metric_name] for new_node_val_metric_single_run in
                     new_node_val_metric_all_runs]
        logger.info(
            f'new node validate {metric_name}, {list_mets}')
        new_node_val_metric[metric_name] = np.mean(list_mets)
        logger.info(
            f'average new node validate {metric_name}, {np.mean(list_mets):.4f} '
            f'± {np.std(list_mets, ddof=1):.4f}')

    average_test_metric, new_node_test_metric = {}, {}
    for metric_name in test_metric_all_runs[0].keys():
        list_mets = [test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]
        logger.info(
            f'test {metric_name}, {list_mets}')
        average_test_metric[metric_name] = np.mean(list_mets)
        logger.info(
            f'average test {metric_name}, {average_test_metric[metric_name]:.4f} '
            f'± {np.std(list_mets, ddof=1):.4f}')

    for metric_name in new_node_test_metric_all_runs[0].keys():
        list_mets = [new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in
                     new_node_test_metric_all_runs]
        logger.info(
            f'new node test {metric_name}, {list_mets}')
        new_node_test_metric[metric_name] = np.mean(list_mets)
        logger.info(
            f'average new node test {metric_name}, {new_node_test_metric[metric_name]:.4f} '
            f'± {np.std(list_mets, ddof=1):.4f}')
    sys.exit()

