from typing import List

from dgl.nn.pytorch import SAGEConv

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

class TBD(nn.Module):
    def __init__(self,
                 num_nodes: int,
                 in_feats: int,
                 time_dim: int,
                 hidden_dim: int,
                 out_dim: int,
                 start_times: List[float],
                 end_times: List[float],
                 device: str = 'cpu',
                 num_gnn_layers: int = 2,
                 dropout: float = 0.1,
                 use_batch: bool = False,
                 batch_size: int = None):
        super().__init__()
        self.num_nodes = num_nodes
        self.out_dim = out_dim
        self.end_times = end_times
        self.start_times = start_times
        self.device = device
        self.use_batch = use_batch
        self.batch_size = batch_size

        self.time_encoder = nn.Embedding(10000, time_dim)  # support at most 10000 blocks

        self.gnn_layers = nn.ModuleList()

        # GNN (use GraphSAGE)
        for i in range(num_gnn_layers):
            in_dim = in_feats + time_dim if i == 0 else hidden_dim
            self.gnn_layers.append(SAGEConv(
                in_dim,
                hidden_dim if i != num_gnn_layers - 1 else out_dim,
                aggregator_type='mean'
            ))
        # GCN
        # for i in range(num_gnn_layers):
        #     in_dim = in_feats + time_dim if i == 0 else hidden_dim
        #     self.gnn_layers.append(GCNConv(
        #         in_dim,
        #         hidden_dim if i!= num_gnn_layers - 1 else out_dim
        #     ))

        self.dropout = nn.Dropout(dropout)

        # MF-GRU
        self.gru = MFGRU(
            input_dim=out_dim,
            hidden_dim=out_dim,
            struct_dim=16,
            device=device
        )


    def forward(self, blocks: List[dgl.DGLGraph]):
        """
        parameter：
            blocks: DGL list
        return：
            representation list [num_blocks, N_i, out_dim]
        """
        block_reprs = []
        num_blocks = len(blocks)

        for block_idx, block in enumerate(blocks):
            time_id = torch.tensor(block_idx + 1, device=self.device)

            block.to(self.device)

            # process
            h = self.process_block(block, time_id)
            h = h.to(self.device)
            block_reprs.append(h)

        h = torch.zeros(self.num_nodes, self.out_dim, device=self.device)
        hidden_states = []

        for b in range(len(blocks)):
            # get features and structure
            x = block_reprs[b]
            current_graph = blocks[b]

            # MFG-GRU
            h = self.gru(x, h, current_graph)
            hidden_states.append(h)


        return hidden_states


    def process_block(self,
                      block: dgl.DGLGraph,
                      time_id: torch.Tensor) -> torch.Tensor:
        """
        process block
        """
        # [N, feat_dim]
        node_feats = block.ndata['feat']
        node_feats = torch.Tensor(node_feats).to(self.device)

        # [N, time_dim]
        time_code = self.time_encoder(
            torch.full((block.num_nodes(),),
                       time_id,
                       device=self.device
                       ))

        # batch
        if self.use_batch and self.batch_size is not None:
            batch_outputs = []
            for nids in torch.split(torch.arange(block.num_nodes()), self.batch_size):
                subg = dgl.node_subgraph(block, nids)
                batch_feats = self._process_single_batch(subg, node_feats[nids], time_code[nids])
                batch_outputs.append(batch_feats)
            h = torch.cat(batch_outputs, dim=0)
        else:
            h = self._process_single_batch(block, node_feats, time_code)

        return h

    def _process_single_batch(self,
                              block: dgl.DGLGraph,
                              node_feats: torch.Tensor,
                              time_code: torch.Tensor) -> torch.Tensor:
        """
        process single batch
        """
        h = torch.cat([node_feats, time_code], dim=-1)
        block = block.to(self.device)
        h = h.to(self.device)

        # GNN
        for i, layer in enumerate(self.gnn_layers):
            h = layer(block, h)
            if i != len(self.gnn_layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)

        return h


class MFGRU(nn.Module):
    """ MF-GRU """

    def __init__(self, input_dim, hidden_dim, struct_dim=16, device='cpu'):
        super().__init__()
        self.struct_encoder = nn.Sequential(
            nn.Linear(1, struct_dim),
            nn.ReLU()
        )

        self.reset_gate = nn.Linear(input_dim + hidden_dim + struct_dim, hidden_dim)
        self.update_gate = nn.Linear(input_dim + hidden_dim + struct_dim, hidden_dim)
        self.candidate = nn.Linear(input_dim + hidden_dim + struct_dim, hidden_dim)

        self.feature_gate = nn.Linear(input_dim + struct_dim, hidden_dim)

        self.device = device

    def forward(self, x, h_prev, graph):
        """
            x: current input [N, input_dim]
            h_prev: prev output [N, hidden_dim]
            graph: block structure
        """

        degrees = graph.in_degrees().float().unsqueeze(-1)  # [N, 1]
        degrees = degrees.to(self.device)

        struct_feat = self.struct_encoder(degrees)  # [N, struct_dim]

        combined = torch.cat([x, h_prev, struct_feat], dim=-1)  # [N, input+hidden+struct]

        # Reset Gate
        r = torch.sigmoid(self.reset_gate(combined))
        # Update Gate
        z = torch.sigmoid(self.update_gate(combined))

        c = torch.tanh(self.candidate(torch.cat([x, r * h_prev, struct_feat], dim=-1)))

        feat_gate = torch.sigmoid(self.feature_gate(torch.cat([x, struct_feat], dim=-1)))

        h_new = (1 - z) * h_prev + z * c
        return feat_gate * h_new