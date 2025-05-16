import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MergeLayer(nn.Module):

    def __init__(self, input_dim1: int, input_dim2: int, input_dim3: int, hidden_dim: int, output_dim: int):
        """
        Merge Layer to merge two inputs via: input_dim1 + input_dim2 -> hidden_dim -> output_dim.
        :param input_dim1: int, dimension of first input
        :param input_dim2: int, dimension of the second input
        :param hidden_dim: int, hidden dimension
        :param output_dim: int, dimension of the output
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim1 + input_dim2 + input_dim3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()

    def forward(self, input_1: torch.Tensor, input_2: torch.Tensor, input_3: torch.Tensor):
        """
        merge and project the inputs
        :param input_1: Tensor, shape (*, input_dim1)
        :param input_2: Tensor, shape (*, input_dim2)
        :return:
        """
        # Tensor, shape (*, input_dim1 + input_dim2)
        if input_2 is None:
            x = input_1
        elif input_3 is None:
            x = torch.cat([input_1, input_2], dim=1)
        else:
            x = torch.cat([input_1, input_2, input_3], dim=1)
        # Tensor, shape (*, output_dim)
        h = self.fc2(self.act(self.fc1(x)))
        return h

