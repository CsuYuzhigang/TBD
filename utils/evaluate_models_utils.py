import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from utils.metrics import get_link_prediction_metrics
from utils.utils import compute_node_smoothness_loss
from utils.utils import NegativeEdgeSampler
from utils.DataLoader import Data


def evaluate(model: nn.Module,
             evaluate_idx_data_loader: DataLoader,
             evaluate_neg_edge_sampler: NegativeEdgeSampler,
             evaluate_data: Data,
             blocks,
             end_times: list,
             smoothness_weight: float,
             device: str):
    """
    evaluate models on the link prediction task
    :param model: nn.Module, the model to be evaluated
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate negative edge sampler
    :param evaluate_data: Data, data to be evaluated
    :return:
    """
    # Ensures the random sampler uses a fixed seed for evaluation (i.e. we always sample the same negatives for validation / test set)
    assert evaluate_neg_edge_sampler.seed is not None
    evaluate_neg_edge_sampler.reset_random_state()

    model.eval()

    with torch.no_grad():
        # store evaluate losses and metrics
        evaluate_losses, evaluate_metrics = [], []

        # Obtain the node representation of each temporal block
        block_reprs_list = model[0](blocks=blocks)

        # progress bar
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)

        node_smoothness_loss = compute_node_smoothness_loss(blocks, block_reprs_list, base_lambda=smoothness_weight).to(device)


        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):

            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids, batch_raw_edge_ids = \
                evaluate_data.src_node_ids[evaluate_data_indices], evaluate_data.dst_node_ids[evaluate_data_indices], \
                    evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices], \
                    evaluate_data.raw_edge_ids[evaluate_data_indices]

            if not isinstance(batch_src_node_ids, np.ndarray):
                # convert to ndarray
                batch_src_node_ids = np.array([batch_src_node_ids])
                batch_dst_node_ids = np.array([batch_dst_node_ids])
                batch_raw_edge_ids = np.array([batch_raw_edge_ids])
                batch_edge_ids = np.array([batch_edge_ids])
                batch_node_interact_times = np.array([batch_node_interact_times])

            # negative sampling
            #  _ -> batch_neg_src_node_ids
            _, batch_neg_dst_node_ids = evaluate_neg_edge_sampler.sample(size=len(batch_src_node_ids),
                                                                         batch_src_node_ids=batch_src_node_ids,
                                                                         batch_dst_node_ids=batch_dst_node_ids,
                                                                         current_batch_start_time=batch_node_interact_times.min(),
                                                                         current_batch_end_time=batch_node_interact_times.max())

            batch_neg_raw_edge_ids = np.zeros_like(batch_raw_edge_ids)
            batch_neg_src_node_ids = batch_src_node_ids

            # get temporal embedding of source and destination nodes
            batch_src_node_embeddings_list, batch_dst_node_embeddings_list, batch_neg_dst_node_embeddings_list = [], [], []
            positive_probabilities_all_list, negative_probabilities_all_list = [], []

            for i, interact_time in enumerate(batch_node_interact_times):
                src_node_embeddings = torch.Tensor([]).to(device)
                dst_node_embeddings = torch.Tensor([]).to(device)
                neg_dst_node_embeddings = torch.Tensor([]).to(device)
                positive_probabilities_all = torch.Tensor([]).to(device)
                negative_probabilities_all = torch.Tensor([]).to(device)

                for j, end_time in enumerate(end_times):
                    if end_time >= interact_time and j > 0:
                        src_node_embeddings = torch.cat(
                            (src_node_embeddings,
                             block_reprs_list[j - 1][batch_src_node_ids[i] - 1].unsqueeze(0)),
                            dim=0)
                        dst_node_embeddings = torch.cat(
                            (dst_node_embeddings,
                             block_reprs_list[j - 1][batch_dst_node_ids[i] - 1].unsqueeze(0)),
                            dim=0)
                        neg_dst_node_embeddings = torch.cat((neg_dst_node_embeddings, block_reprs_list[j - 1][
                            batch_neg_dst_node_ids[i] - 1].unsqueeze(0)), dim=0)
                        break

                batch_src_node_embeddings_list.append(src_node_embeddings)
                batch_dst_node_embeddings_list.append(dst_node_embeddings)
                batch_neg_dst_node_embeddings_list.append(neg_dst_node_embeddings)


            positive_probabilities = torch.Tensor([]).to(device)
            negative_probabilities = torch.Tensor([]).to(device)

            for batch_src_node_embeddings, batch_dst_node_embeddings, batch_neg_dst_node_embeddings \
                    in zip(batch_src_node_embeddings_list, batch_dst_node_embeddings_list,
                           batch_neg_dst_node_embeddings_list):
                positive_probability = model[1](input_1=batch_src_node_embeddings,
                                                input_2=batch_dst_node_embeddings,
                                                input_3=None).squeeze(dim=-1).sigmoid()
                negative_probability = model[1](input_1=batch_src_node_embeddings,
                                                input_2=batch_neg_dst_node_embeddings,
                                                input_3=None).squeeze(dim=-1).sigmoid()

                positive_probability = positive_probability.to(device)
                negative_probability = negative_probability.to(device)

                positive_probabilities = torch.cat((positive_probabilities, positive_probability.unsqueeze(0)),
                                                   dim=0)
                negative_probabilities = torch.cat((negative_probabilities, negative_probability.unsqueeze(0)),
                                                   dim=0)

            predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
            labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)],
                               dim=0)  # (0, 1)


            loss = nn.BCELoss()(input=predicts, target=labels) + node_smoothness_loss

            evaluate_losses.append(loss.item())

            evaluate_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))

            evaluate_idx_data_loader_tqdm.set_description(
                f'evaluate for the {batch_idx + 1}-th batch, evaluate loss: {loss.item()}')

    return evaluate_losses, evaluate_metrics

def evaluate_mixture(model: nn.Module,
                     evaluate_idx_data_loader: DataLoader,
                     evaluate_neg_edge_sampler: NegativeEdgeSampler,
                     evaluate_data: Data,
                     blocks,
                     end_times: list,
                     smoothness_weight: float,
                     device: str):
    """
    "mixture temporal blocks prediction" evaluate models on the link prediction task
    :param model: nn.Module, the model to be evaluated
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate negative edge sampler
    :param evaluate_data: Data, data to be evaluated
    :return:
    """
    # Ensures the random sampler uses a fixed seed for evaluation (i.e. we always sample the same negatives for validation / test set)
    assert evaluate_neg_edge_sampler.seed is not None
    evaluate_neg_edge_sampler.reset_random_state()

    model.eval()

    with torch.no_grad():
        # store evaluate losses and metrics
        evaluate_losses, evaluate_metrics = [], []

        block_reprs_list = model[0](blocks=blocks)


        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)

        node_smoothness_loss = compute_node_smoothness_loss(blocks, block_reprs_list, base_lambda=smoothness_weight).to(device)  # 能量损失


        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):

            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids, batch_raw_edge_ids = \
                evaluate_data.src_node_ids[evaluate_data_indices], evaluate_data.dst_node_ids[evaluate_data_indices], \
                    evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices], \
                    evaluate_data.raw_edge_ids[evaluate_data_indices]

            if not isinstance(batch_src_node_ids, np.ndarray):
                # convert to ndarray
                batch_src_node_ids = np.array([batch_src_node_ids])
                batch_dst_node_ids = np.array([batch_dst_node_ids])
                batch_raw_edge_ids = np.array([batch_raw_edge_ids])
                batch_edge_ids = np.array([batch_edge_ids])
                batch_node_interact_times = np.array([batch_node_interact_times])

            # negative sampling
            #  _ -> batch_neg_src_node_ids
            _, batch_neg_dst_node_ids = evaluate_neg_edge_sampler.sample(size=len(batch_src_node_ids),
                                                                         batch_src_node_ids=batch_src_node_ids,
                                                                         batch_dst_node_ids=batch_dst_node_ids,
                                                                         current_batch_start_time=batch_node_interact_times.min(),
                                                                         current_batch_end_time=batch_node_interact_times.max())

            batch_neg_raw_edge_ids = np.zeros_like(batch_raw_edge_ids)
            batch_neg_src_node_ids = batch_src_node_ids

            # get temporal embedding of source and destination nodes
            batch_src_node_embeddings_list, batch_dst_node_embeddings_list, batch_neg_dst_node_embeddings_list = [], [], []
            # positive_probabilities_all_list, negative_probabilities_all_list = [], []
            time_weights_list = []

            for i, interact_time in enumerate(batch_node_interact_times):
                src_node_embeddings = torch.Tensor([]).to(device)
                dst_node_embeddings = torch.Tensor([]).to(device)
                neg_dst_node_embeddings = torch.Tensor([]).to(device)
                time_weights = torch.Tensor([]).to(device)

                for j, end_time in enumerate(end_times):
                    if end_time > interact_time:
                        break
                    src_node_embeddings = torch.cat(
                        (src_node_embeddings, block_reprs_list[j][batch_src_node_ids[i] - 1].unsqueeze(0)), dim=0)
                    dst_node_embeddings = torch.cat(
                        (dst_node_embeddings, block_reprs_list[j][batch_dst_node_ids[i] - 1].unsqueeze(0)), dim=0)
                    neg_dst_node_embeddings = torch.cat(
                        (neg_dst_node_embeddings, block_reprs_list[j][batch_neg_dst_node_ids[i] - 1].unsqueeze(0)),
                        dim=0)
                    # positive_probabilities_all = torch.cat((positive_probabilities_all,
                    #                                         probability_matrix_list[j][
                    #                                             batch_src_node_ids[i] - 1,
                    #                                             batch_dst_node_ids[i] - 1].unsqueeze(0)),
                    #                                        dim=0)
                    # negative_probabilities_all = torch.cat((negative_probabilities_all,
                    #                                         probability_matrix_list[j][
                    #                                             batch_src_node_ids[i] - 1,
                    #                                             batch_neg_dst_node_ids[i] - 1].unsqueeze(
                    #                                             0)),
                    #                                        dim=0)
                    time_weights = torch.cat((time_weights, torch.tensor([end_time - interact_time], device=device).unsqueeze(0)),
                                             dim=0)

                if time_weights.shape[0] == 0:
                    src_node_embeddings = block_reprs_list[0][batch_src_node_ids[i] - 1].unsqueeze(0)
                    dst_node_embeddings = block_reprs_list[0][batch_dst_node_ids[i] - 1].unsqueeze(0)
                    neg_dst_node_embeddings = block_reprs_list[0][batch_neg_dst_node_ids[i] - 1].unsqueeze(0)
                    # positive_probabilities_all = torch.cat((positive_probabilities_all,
                    #                                         probability_matrix_list[0][
                    #                                             batch_src_node_ids[i] - 1,
                    #                                             batch_dst_node_ids[i] - 1].unsqueeze(0)),
                    #                                        dim=0)
                    # negative_probabilities_all = torch.cat((negative_probabilities_all,
                    #                                         probability_matrix_list[0][
                    #                                             batch_src_node_ids[i] - 1,
                    #                                             batch_neg_dst_node_ids[i] - 1].unsqueeze(
                    #                                             0)),
                    #                                        dim=0)
                    time_weights = torch.tensor([end_times[0] - interact_time]).unsqueeze(0)

                if not time_weights.sum().item() == 0:
                    time_weights = time_weights / abs(time_weights.sum())  # normalize


                batch_src_node_embeddings_list.append(src_node_embeddings)
                batch_dst_node_embeddings_list.append(dst_node_embeddings)
                batch_neg_dst_node_embeddings_list.append(neg_dst_node_embeddings)
                # positive_probabilities_all_list.append(positive_probabilities_all)
                # negative_probabilities_all_list.append(negative_probabilities_all)
                time_weights_list.append(time_weights)


            positive_probabilities = torch.Tensor([]).to(device)
            negative_probabilities = torch.Tensor([]).to(device)

            for batch_src_node_embeddings, batch_dst_node_embeddings, batch_neg_dst_node_embeddings, time_weights \
                    in zip(batch_src_node_embeddings_list, batch_dst_node_embeddings_list, batch_neg_dst_node_embeddings_list, time_weights_list):
            # for batch_positive_probabilities_all, batch_negative_probabilities_all, time_weights \
            #         in zip(positive_probabilities_all_list, negative_probabilities_all_list, time_weights_list):
                positive_probabilities_all = model[1](input_1=batch_src_node_embeddings,
                                                      input_2=batch_dst_node_embeddings,
                                                      input_3=None).squeeze(dim=-1).sigmoid()
                negative_probabilities_all = model[1](input_1=batch_src_node_embeddings,
                                                      input_2=batch_neg_dst_node_embeddings,
                                                      input_3=None).squeeze(dim=-1).sigmoid()
                # Softmax
                softmax_weights = torch.softmax(time_weights, dim=0)  # (N, 1)
                softmax_weights = softmax_weights.squeeze(dim=-1)

                positive_probabilities_all = positive_probabilities_all.to(device)
                negative_probabilities_all = negative_probabilities_all.to(device)
                softmax_weights = softmax_weights.to(device)

                positive_probability = torch.sum(softmax_weights * positive_probabilities_all)
                negative_probability = torch.sum(softmax_weights * negative_probabilities_all)

                positive_probabilities = torch.cat((positive_probabilities, positive_probability.unsqueeze(0)), dim=0)
                negative_probabilities = torch.cat((negative_probabilities, negative_probability.unsqueeze(0)), dim=0)

                # # Softmax
                # softmax_weights = torch.softmax(time_weights, dim=0)  # (N, 1)
                # softmax_weights = softmax_weights.squeeze(dim=-1)
                #
                # batch_positive_probabilities_all = batch_positive_probabilities_all.to(device)
                # batch_positive_probabilities_all = batch_positive_probabilities_all.to(device)
                # softmax_weights = softmax_weights.to(device)
                #
                # positive_probability = torch.sum(softmax_weights * batch_positive_probabilities_all)
                # negative_probability = torch.sum(softmax_weights * batch_positive_probabilities_all)
                #
                # positive_probabilities = torch.cat((positive_probabilities, positive_probability.unsqueeze(0)),
                #                                    dim=0)
                # negative_probabilities = torch.cat((negative_probabilities, negative_probability.unsqueeze(0)),
                #                                    dim=0)

            predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
            labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)],
                               dim=0)  # (0, 1)


            loss = nn.BCELoss()(input=predicts, target=labels) + node_smoothness_loss

            evaluate_losses.append(loss.item())

            evaluate_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))

            evaluate_idx_data_loader_tqdm.set_description(
                f'evaluate for the {batch_idx + 1}-th batch, evaluate loss: {loss.item()}')

    return evaluate_losses, evaluate_metrics

