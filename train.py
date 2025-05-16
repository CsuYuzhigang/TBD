import logging
import sys
import os
import time

from tqdm import tqdm
import numpy as np
import warnings
import json
import torch
import torch.nn as nn
# import wandb

from models.TBD import TBD
from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes, create_optimizer, \
    get_historical_neighbor_sampler, create_temporal_blocks, compute_node_smoothness_loss
from utils.utils import get_neighbor_sampler, NegativeEdgeSampler
from utils.evaluate_models_utils import evaluate, evaluate_mixture
from utils.metrics import get_link_prediction_metrics
from utils.DataLoader import get_idx_data_loader, get_link_prediction_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_prediction_args
from models.modules import MergeLayer

if __name__ == "__main__":


    warnings.filterwarnings('ignore')
    torch.set_num_threads(1)

    # get arguments
    args = get_link_prediction_args(is_evaluation=False)

    # get data for training, validation and testing
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = \
        get_link_prediction_data(dataset_name=args.dataset_name, val_ratio=args.val_ratio, test_ratio=args.test_ratio)
    node_raw_features = node_raw_features[1:, :]
    edge_raw_features = edge_raw_features[1:, :]


    train_neighbor_sampler = get_neighbor_sampler(data=train_data,
                                                  sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                  time_scaling_factor=args.time_scaling_factor, seed=0)

    historical_train_neighbor_sample = get_historical_neighbor_sampler(data=train_data,
                                                                       sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                                       time_scaling_factor=args.time_scaling_factor,
                                                                       seed=1)


    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                 time_scaling_factor=args.time_scaling_factor, seed=1)

    full_edge_neighbor_sampler = get_historical_neighbor_sampler(data=full_data,
                                                                 sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                                 time_scaling_factor=args.time_scaling_factor, seed=1)

    # initialize negative samplers, set seeds for validation and testing so negatives are the same across different runs
    # in the inductive setting, negatives are sampled only amongst other new nodes
    # train negative edge sampler does not need to specify the seed, but evaluation samplers need to do so
    if args.negative_sample_strategy == 'random':

        train_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=train_data.src_node_ids,
                                                     dst_node_ids=train_data.dst_node_ids,
                                                     negative_sample_strategy=args.negative_sample_strategy)

        val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids,
                                                   dst_node_ids=full_data.dst_node_ids,
                                                   negative_sample_strategy=args.negative_sample_strategy,
                                                   seed=0)

        new_node_val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_val_data.src_node_ids,
                                                            dst_node_ids=new_node_val_data.dst_node_ids,
                                                            negative_sample_strategy=args.negative_sample_strategy,
                                                            seed=1)

        test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids,
                                                    dst_node_ids=full_data.dst_node_ids,
                                                    negative_sample_strategy=args.negative_sample_strategy,
                                                    seed=2)

        new_node_test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_test_data.src_node_ids,
                                                             dst_node_ids=new_node_test_data.dst_node_ids,
                                                             negative_sample_strategy=args.negative_sample_strategy,
                                                             seed=3)
    else:
        train_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids,
                                                     dst_node_ids=full_data.dst_node_ids,
                                                     interact_times=full_data.node_interact_times,
                                                     last_observed_time=train_data.node_interact_times[-1],
                                                     negative_sample_strategy=args.negative_sample_strategy, seed=4)
        val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids,
                                                   dst_node_ids=full_data.dst_node_ids,
                                                   interact_times=full_data.node_interact_times,
                                                   last_observed_time=train_data.node_interact_times[-1],
                                                   negative_sample_strategy=args.negative_sample_strategy, seed=0)
        new_node_val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_val_data.src_node_ids,
                                                            dst_node_ids=new_node_val_data.dst_node_ids,
                                                            interact_times=new_node_val_data.node_interact_times,
                                                            last_observed_time=train_data.node_interact_times[-1],
                                                            negative_sample_strategy=args.negative_sample_strategy,
                                                            seed=1)
        test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids,
                                                    dst_node_ids=full_data.dst_node_ids,
                                                    interact_times=full_data.node_interact_times,
                                                    last_observed_time=val_data.node_interact_times[-1],
                                                    negative_sample_strategy=args.negative_sample_strategy, seed=2)
        new_node_test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_test_data.src_node_ids,
                                                             dst_node_ids=new_node_test_data.dst_node_ids,
                                                             interact_times=new_node_test_data.node_interact_times,
                                                             last_observed_time=val_data.node_interact_times[-1],
                                                             negative_sample_strategy=args.negative_sample_strategy,
                                                             seed=3)

    # get data loaders
    train_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(train_data.src_node_ids))),
                                                batch_size=args.batch_size, shuffle=False)
    val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.src_node_ids))),
                                              batch_size=args.batch_size, shuffle=False)
    new_node_val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_val_data.src_node_ids))),
                                                       batch_size=args.batch_size, shuffle=False)
    test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.src_node_ids))),
                                               batch_size=args.batch_size, shuffle=False)
    new_node_test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_test_data.src_node_ids))),
                                                        batch_size=args.batch_size, shuffle=False)

    # Divide into blocks
    blocks, start_times, end_times = create_temporal_blocks(dataset_name=args.dataset_name,
                                                            node_raw_features=node_raw_features,
                                                            edge_raw_features=edge_raw_features,
                                                            full_data=full_data,
                                                            stride=args.stride,
                                                            block_size=args.block_size,
                                                            device=args.device,
                                                            drop_last=False)

    evaluate_func = evaluate
    if args.dataset_name in ['USLegis', 'enron']:
        evaluate_func = evaluate_mixture  # mixture temporal blocks prediction

    # blocks = convert_to_gpu(blocks, device=args.device)
    # start_times = convert_to_gpu(start_times, device=args.device)
    # end_times = convert_to_gpu(end_times, device=args.device)


    val_metric_all_runs, new_node_val_metric_all_runs, test_metric_all_runs, new_node_test_metric_all_runs = [], [], [], []

    for run in range(args.start_s, args.num_runs):  # run several times

        set_random_seed(seed=run)

        args.seed = run
        args.save_model_name = f'{args.model_name}_seed{args.seed}'

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/", exist_ok=True)

        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(
            f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/{str(time.time())}.log")
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
        # config_saver = ConfigSaving(args.model_name,
        #                             f"dataset_{args.dataset_name}_"
        #                             f"seed_{args.seed}_train_"
        #                             f"linkPred",
        #                             args, run_dir)

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

        optimizer = create_optimizer(model=model, optimizer_name=args.optimizer, learning_rate=args.learning_rate,
                                     weight_decay=args.weight_decay)

        model = convert_to_gpu(model, device=args.device)

        save_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.prefix_name}_{args.save_model_name}/"
        # shutil.rmtree(save_model_folder, ignore_errors=True)
        os.makedirs(save_model_folder, exist_ok=True)

        # 早停
        early_stopping = EarlyStopping(patience=args.patience, save_model_folder=save_model_folder,
                                       save_model_name=args.save_model_name, logger=logger, model_name=args.model_name)


        loss_func = nn.BCELoss()
        block_reprs_list = None

        # torch.autograd.set_detect_anomaly(True)

        for epoch in range(args.num_epochs):
            model.train()
            # training, only use training graph
            # model[0].set_neighbor_sampler(train_neighbor_sampler)

            # store train losses and metrics
            train_losses, train_metrics = [], []

            # loss = torch.tensor([0.0]).to(args.device)

            # progress bar
            train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120)


            for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):

                # Obtain the node representation of each temporal block
                block_reprs_list = model[0](blocks=blocks)

                batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids, batch_raw_edge_ids = \
                    train_data.src_node_ids[train_data_indices], train_data.dst_node_ids[train_data_indices], \
                        train_data.node_interact_times[train_data_indices], train_data.edge_ids[train_data_indices], \
                        train_data.raw_edge_ids[train_data_indices]

                if not isinstance(batch_src_node_ids, np.ndarray):
                    # convert to ndarray
                    batch_src_node_ids = np.array([batch_src_node_ids])
                    batch_dst_node_ids = np.array([batch_dst_node_ids])
                    batch_raw_edge_ids = np.array([batch_raw_edge_ids])
                    batch_edge_ids = np.array([batch_edge_ids])
                    batch_node_interact_times = np.array([batch_node_interact_times])

                # negative sampling
                #  _ -> batch_neg_src_node_ids
                _, batch_neg_dst_node_ids = train_neg_edge_sampler.sample(size=len(batch_src_node_ids),
                                                                          batch_src_node_ids=batch_src_node_ids,
                                                                          batch_dst_node_ids=batch_dst_node_ids,
                                                                          current_batch_start_time=batch_node_interact_times.min(),
                                                                          current_batch_end_time=batch_node_interact_times.max())

                batch_neg_raw_edge_ids = np.zeros_like(batch_raw_edge_ids)  # Set the negative sample ID to 0 (indicating no actual edge).
                batch_neg_src_node_ids = batch_src_node_ids


                # get temporal embedding of source and destination nodes
                batch_src_node_embeddings_list, batch_dst_node_embeddings_list, batch_neg_dst_node_embeddings_list = [], [], []
                positive_probabilities_all_list, negative_probabilities_all_list = [], []
                time_weights_list = []


                for i, interact_time in enumerate(batch_node_interact_times):
                    src_node_embeddings = torch.Tensor([]).to(args.device)
                    dst_node_embeddings = torch.Tensor([]).to(args.device)
                    neg_dst_node_embeddings = torch.Tensor([]).to(args.device)
                    positive_probabilities_all = torch.Tensor([]).to(args.device)
                    negative_probabilities_all = torch.Tensor([]).to(args.device)

                    for j, end_time in enumerate(end_times):
                        if end_time >= interact_time and j > 0:

                            src_node_embeddings = torch.cat(
                                (src_node_embeddings, block_reprs_list[j - 1][batch_src_node_ids[i] - 1].unsqueeze(0)),
                                dim=0)
                            dst_node_embeddings = torch.cat(
                                (dst_node_embeddings, block_reprs_list[j - 1][batch_dst_node_ids[i] - 1].unsqueeze(0)),
                                dim=0)
                            neg_dst_node_embeddings = torch.cat((neg_dst_node_embeddings, block_reprs_list[j - 1][
                                batch_neg_dst_node_ids[i] - 1].unsqueeze(0)), dim=0)
                            break

                    batch_src_node_embeddings_list.append(src_node_embeddings)
                    batch_dst_node_embeddings_list.append(dst_node_embeddings)
                    batch_neg_dst_node_embeddings_list.append(neg_dst_node_embeddings)

                if args.dataset_name in ['USLegis', 'enron']:
                    # mixture temporal blocks prediction
                    batch_src_node_embeddings_list, batch_dst_node_embeddings_list, batch_neg_dst_node_embeddings_list = [], [], []
                    positive_probabilities_all_list, negative_probabilities_all_list = [], []
                    time_weights_list = []


                    for i, interact_time in enumerate(batch_node_interact_times):
                        src_node_embeddings = torch.Tensor([]).to(args.device)
                        dst_node_embeddings = torch.Tensor([]).to(args.device)
                        neg_dst_node_embeddings = torch.Tensor([]).to(args.device)
                        positive_probabilities_all = torch.Tensor([]).to(args.device)
                        negative_probabilities_all = torch.Tensor([]).to(args.device)
                        time_weights = torch.Tensor([]).to(args.device)

                        for j, end_time in enumerate(end_times):
                            if end_time > interact_time:
                                break

                            src_node_embeddings = torch.cat(
                                (src_node_embeddings, block_reprs_list[j][batch_src_node_ids[i] - 1].unsqueeze(0)),
                                dim=0)
                            dst_node_embeddings = torch.cat(
                                (dst_node_embeddings, block_reprs_list[j][batch_dst_node_ids[i] - 1].unsqueeze(0)),
                                dim=0)
                            neg_dst_node_embeddings = torch.cat((neg_dst_node_embeddings, block_reprs_list[j][
                                batch_neg_dst_node_ids[i] - 1].unsqueeze(0)), dim=0)
                            time_weights = torch.cat(
                                (time_weights,
                                 torch.tensor([end_time - interact_time], device=args.device).unsqueeze(0)), dim=0)

                        if time_weights.shape[0] == 0:
                            src_node_embeddings = block_reprs_list[0][batch_src_node_ids[i] - 1].unsqueeze(0)
                            dst_node_embeddings = block_reprs_list[0][batch_dst_node_ids[i] - 1].unsqueeze(0)
                            neg_dst_node_embeddings = block_reprs_list[0][batch_neg_dst_node_ids[i] - 1].unsqueeze(0)
                            time_weights = torch.tensor([end_times[0] - interact_time], device=args.device).unsqueeze(0)

                        if not time_weights.sum().item() == 0:
                            time_weights = time_weights / abs(time_weights.sum())  # normalize

                        batch_src_node_embeddings_list.append(src_node_embeddings)
                        batch_dst_node_embeddings_list.append(dst_node_embeddings)
                        batch_neg_dst_node_embeddings_list.append(neg_dst_node_embeddings)
                        time_weights_list.append(time_weights)


                # Calculate the probability of link prediction
                positive_probabilities = torch.Tensor([]).to(args.device)
                negative_probabilities = torch.Tensor([]).to(args.device)
                if args.dataset_name not in ['USLegis', 'enron']:
                    for batch_src_node_embeddings, batch_dst_node_embeddings, batch_neg_dst_node_embeddings \
                            in zip(batch_src_node_embeddings_list, batch_dst_node_embeddings_list,
                                   batch_neg_dst_node_embeddings_list):
                        positive_probability = model[1](input_1=batch_src_node_embeddings,
                                                              input_2=batch_dst_node_embeddings,
                                                              input_3=None).squeeze(dim=-1).sigmoid()
                        negative_probability = model[1](input_1=batch_src_node_embeddings,
                                                              input_2=batch_neg_dst_node_embeddings,
                                                              input_3=None).squeeze(dim=-1).sigmoid()

                        positive_probability = positive_probability.to(args.device)
                        negative_probability = negative_probability.to(args.device)

                        positive_probabilities = torch.cat((positive_probabilities, positive_probability.unsqueeze(0)),
                                                           dim=0)
                        negative_probabilities = torch.cat((negative_probabilities, negative_probability.unsqueeze(0)),
                                                           dim=0)
                else:
                    # mixture temporal blocks prediction
                    positive_probabilities = torch.Tensor([]).to(args.device)
                    negative_probabilities = torch.Tensor([]).to(args.device)

                    for batch_src_node_embeddings, batch_dst_node_embeddings, batch_neg_dst_node_embeddings, time_weights \
                            in zip(batch_src_node_embeddings_list, batch_dst_node_embeddings_list,
                                   batch_neg_dst_node_embeddings_list, time_weights_list):
                        positive_probabilities_all = model[1](input_1=batch_src_node_embeddings,
                                                              input_2=batch_dst_node_embeddings,
                                                              input_3=None).squeeze(dim=-1).sigmoid()
                        negative_probabilities_all = model[1](input_1=batch_src_node_embeddings,
                                                              input_2=batch_neg_dst_node_embeddings,
                                                              input_3=None).squeeze(dim=-1).sigmoid()

                        # Softmax
                        softmax_weights = torch.softmax(time_weights, dim=0)  # (N, 1)
                        softmax_weights = softmax_weights.squeeze(dim=-1)

                        positive_probabilities_all = positive_probabilities_all.to(args.device)
                        negative_probabilities_all = negative_probabilities_all.to(args.device)
                        softmax_weights = softmax_weights.to(args.device)

                        positive_probability = torch.sum(softmax_weights * positive_probabilities_all)
                        negative_probability = torch.sum(softmax_weights * negative_probabilities_all)

                        positive_probabilities = torch.cat((positive_probabilities, positive_probability.unsqueeze(0)),
                                                           dim=0)
                        negative_probabilities = torch.cat((negative_probabilities, negative_probability.unsqueeze(0)),
                                                           dim=0)


                predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
                labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)],
                                   dim=0)  # labels (0, 1)


                node_smoothness_loss = compute_node_smoothness_loss(blocks, block_reprs_list, base_lambda=args.smoothness_weight).to(args.device)


                loss = loss_func(input=predicts, target=labels) + node_smoothness_loss

                train_losses.append(loss.item())

                train_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))

                train_idx_data_loader_tqdm.set_description(
                    f'Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss.item()}')

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

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

            logger.info(
                f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {np.mean(train_losses):.4f}')

            train_total_metrics, valid_total_metrics, new_node_valid_total_metrics = {}, {}, {}

            for metric_name in train_metrics[0].keys():
                train_total_metrics[metric_name] = np.mean(
                    [train_metric[metric_name] for train_metric in train_metrics])
                logger.info(f'train {metric_name}, {train_total_metrics[metric_name]:.4f}')
            logger.info(f'validate loss: {np.mean(val_losses):.4f}')
            for metric_name in val_metrics[0].keys():
                valid_total_metrics[metric_name] = np.mean([val_metric[metric_name] for val_metric in val_metrics])
                logger.info(f'validate {metric_name}, {valid_total_metrics[metric_name]:.4f}')
            logger.info(f'new node validate loss: {np.mean(new_node_val_losses):.4f}')
            for metric_name in new_node_val_metrics[0].keys():
                new_node_valid_total_metrics[metric_name] = np.mean(
                    [new_node_val_metric[metric_name] for new_node_val_metric in new_node_val_metrics])
                logger.info(f'new node validate {metric_name}, {new_node_valid_total_metrics[metric_name]:.4f}')

            # record the performance
            # config_saver.record({"loss": np.mean(train_losses)}, 0)
            # config_saver.record({"loss": np.mean(val_losses)}, 1)
            # config_saver.record({"new node loss": np.mean(new_node_val_losses)}, 1)
            #
            # config_saver.record(train_total_metrics, 0)
            # config_saver.record(valid_total_metrics, 1)
            # config_saver.record(new_node_valid_total_metrics, pre_fix='new_node', stage=1)

            # perform testing once after test_interval_epochs
            if (epoch + 1) % args.test_interval_epochs == 0:
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

                new_node_test_total_metrics, test_total_metrics = {}, {}

                logger.info(f'test loss: {np.mean(test_losses):.4f}')
                for metric_name in test_metrics[0].keys():
                    test_total_metrics[metric_name] = np.mean(
                        [test_metric[metric_name] for test_metric in test_metrics])
                    logger.info(f'test {metric_name}, {test_total_metrics[metric_name]:.4f}')
                logger.info(f'new node test loss: {np.mean(new_node_test_losses):.4f}')
                for metric_name in new_node_test_metrics[0].keys():
                    new_node_test_total_metrics[metric_name] = np.mean(
                        [new_node_test_metric[metric_name] for new_node_test_metric in new_node_test_metrics])
                    logger.info(f'new node test {metric_name}, {new_node_test_total_metrics[metric_name]:.4f}')

                # config_saver.record({"loss": np.mean(test_losses)}, 2)
                # config_saver.record({"new node loss": np.mean(new_node_val_losses)}, 2)
                #
                # config_saver.record(test_total_metrics, 1)
                # config_saver.record(new_node_test_total_metrics, pre_fix='new_node', stage=2)

            # select the best model based on all the validate metrics
            # If there is no improvement for consecutive patience rounds, then terminate the training prematurely.
            val_metric_indicator = []
            for metric_name in val_metrics[0].keys():
                val_metric_indicator.append(
                    (metric_name, np.mean([val_metric[metric_name] for val_metric in val_metrics]), True))
            early_stop = early_stopping.step(val_metric_indicator, model)

            if early_stop:
                break

        # load the best model
        early_stopping.load_checkpoint(model)

        # evaluate_mixture the best model
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
            average_new_node_val_metric = np.mean(
                [new_node_val_metric[metric_name] for new_node_val_metric in new_node_val_metrics])
            logger.info(f'new node validate {metric_name}, {average_new_node_val_metric:.4f}')
            new_node_val_metric_dict[metric_name] = average_new_node_val_metric

        logger.info(f'test loss: {np.mean(test_losses):.4f}')
        for metric_name in test_metrics[0].keys():
            average_test_metric = np.mean([test_metric[metric_name] for test_metric in test_metrics])
            logger.info(f'test {metric_name}, {average_test_metric:.4f}')
            test_metric_dict[metric_name] = average_test_metric

        logger.info(f'new node test loss: {np.mean(new_node_test_losses):.4f}')
        for metric_name in new_node_test_metrics[0].keys():
            average_new_node_test_metric = np.mean(
                [new_node_test_metric[metric_name] for new_node_test_metric in new_node_test_metrics])
            logger.info(f'new node test {metric_name}, {average_new_node_test_metric:.4f}')
            new_node_test_metric_dict[metric_name] = average_new_node_test_metric



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
            "validate metrics": {metric_name: f'{val_metric_dict[metric_name]:.4f}' for metric_name in
                                 val_metric_dict},
            "new node validate metrics": {metric_name: f'{new_node_val_metric_dict[metric_name]:.4f}' for
                                          metric_name in new_node_val_metric_dict},
            "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in
                             test_metric_dict},
            "new node test metrics": {metric_name: f'{new_node_test_metric_dict[metric_name]:.4f}' for metric_name
                                      in new_node_test_metric_dict}
        }
        # config_saver.record(val_metric_dict, 1)
        # config_saver.record(new_node_val_metric_dict, pre_fix='new_node', stage=1)
        # config_saver.record(test_metric_dict, 2)
        # config_saver.record(new_node_test_metric_dict, pre_fix='new_node', stage=2)

        result_json = json.dumps(result_json, indent=4)

        save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}"
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(save_result_folder, f"{args.save_model_name}.json")

        with open(save_result_path, 'w') as file:
            file.write(result_json)
        # config_saver.end()
    # store the average metrics at the log of the last run
    logger.info(f'metrics over {args.num_runs} runs:')

    # add wandb logInfo
    # run_dir = Path(
    #     f"../results/")
    # if not run_dir.exists():
    #     os.makedirs(str(run_dir))
    args.seed = -1
    # config_saver = ConfigSaving(args.model_name,
    #                             f"dataset_{args.dataset_name}_"
    #                             f"average_train_"
    #                             f"linkPred",
    #                             args, run_dir)

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
    # config_saver.record(average_val_metric, 1)
    # config_saver.record(new_node_val_metric, 1, pre_fix='new_node')

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
    # config_saver.record(average_test_metric, 2)
    # config_saver.record(new_node_test_metric, 2, pre_fix='new_node')
    sys.exit()
