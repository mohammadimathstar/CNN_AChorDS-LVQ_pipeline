import torch.nn as nn
from src.AChorDSLVQ.prototypes_gradients import *
from src.utils.grassmann import init_randn

import torch

from src.AChorDSLVQ.prototypes_gradients import DistanceLayer

class PrototypeLayer(nn.Module):
    def __init__(self,
                 config,
                 dtype=torch.float32,
                 device='cpu'
                ):
        """
        Initialize the PrototypeLayer.

        Args:
            config (dict): Configuration dictionary.
            dtype (torch.dtype, optional): Data type of the tensors. Defaults to torch.float32.
            device (str, optional): Device to use ('cpu' or 'cuda'). Defaults to 'cpu'.
        """
        super().__init__()
        # self._feature_dim = feature_dim
        # self._subspace_dim = subspace_dim
        self._num_prototypes = config['train']['hyperparams']['num_prototypes']
        self._prototype_depth = config['train']['hyperparams']['prototype_depth']
        self._subspace_dim = config['train']['hyperparams']['subspace_dim']
        # self._metric_type = metric_type

        # Initialize prototypes
        self.xprotos, self.yprotos, self.yprotos_mat, self.yprotos_comp_mat = init_randn(
            self._prototype_depth,
            self._subspace_dim,
            num_of_protos=self._num_prototypes,
            num_of_classes=config['data_loader']['num_class'],
            device=device,
        )

        self._number_of_prototypes = self.yprotos.shape[0]

        # Initialize relevance parameters
        self.relevances = nn.Parameter(
            torch.ones((1, self.xprotos.shape[-1]), dtype=dtype, device=device) / self.xprotos.shape[-1]
        )

        self.distance_layer = DistanceLayer



    def forward(self, xs_subspace: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PrototypeLayer.

        Args:
            xs_subspace (torch.Tensor): Input subspaces.

        Returns:
            torch.Tensor: Output from the GeodesicPrototypeLayer.
        """

        return self.distance_layer.apply(
            xs_subspace,
            self.xprotos,
            self.relevances
        )


