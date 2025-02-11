
import numpy as np
import os

import torch
import torch.nn as nn

from src.utils.grassmann import grassmann_repr, grassmann_repr_full, compute_distances_on_grassmann_mdf
from src.utils.glvq import *
from src.AChorDSLVQ.prototypes import PrototypeLayer



LOW_BOUND_LAMBDA = 0.001


class Model(nn.Module):
    def __init__(self,
                 config: dict,
                 feature_extractor: torch.nn.Module,                 
                 add_on_layers: nn.Module = nn.Identity(),
                 device='cpu'
                 ):
        super().__init__()

        self._num_classes = config['data_loader']['num_class']        

        # Set the feature extraction network
        self.feature_extractor = feature_extractor
        self._add_on = add_on_layers


        # Create the prototype layers
        self.prototype_layer = PrototypeLayer(
            config,
            device=device,
        )

    @property
    def prototypes_require_grad(self) -> bool:
        return self.prototype_layer.xprotos.requires_grad

    @prototypes_require_grad.setter
    def prototypes_require_grad(self, val: bool):
        self.prototype_layer.xprotos.requires_grad = val

    @property
    def features_require_grad(self) -> bool:
        return any([param.requires_grad for param in self._net.parameters()])

    @features_require_grad.setter
    def features_require_grad(self, val: bool):
        for param in self._net.parameters():
            param.requires_grad = val

    @property
    def add_on_layers_require_grad(self) -> bool:
        return any([param.requires_grad for param in self._add_on.parameters()])

    @add_on_layers_require_grad.setter
    def add_on_layers_require_grad(self, val: bool):
        for param in self._add_on.parameters():
            param.requires_grad = val

    def forward(self,
                inputs: torch.Tensor,
                **kwargs):
        """
        Forward pass of the model.

        Args:
            inputs (torch.Tensor): A batch of input data.

        Returns:
            tuple: Distances and Qw.
        """
        # Perform forward pass through the feature extractor
        features = self.feature_extractor(inputs)
        features = self._add_on(features)

        # Get Grassmann representation of the features
        subspaces = grassmann_repr(features, self.prototype_layer._subspace_dim)

        # Compute distance between subspaces and prototypes
        distance, Qw = self.prototype_layer(
            subspaces)  # SHAPE: (batch_size, num_prototypes, D: dim_of_data, d: dim_of_subspace)

        return distance, Qw

    def forward_partial(self,
                        inputs: torch.Tensor):
        # Perform forward pass through the feature extractor
        features = self.feature_extractor(inputs)
        features = self._add_on(features)

        # Get Grassmann representation of the features
        subspaces, Vh, S = grassmann_repr_full(features, self.prototype_layer._subspace_dim)

        output = compute_distances_on_grassmann_mdf(
            subspaces,
            self.prototype_layer.xprotos,
            self.prototype_layer.relevances,
        )
        # ypred = self.prototype_layer.yprotos[output['distance'].argmin(axis=1)]

        return features, subspaces, Vh, S, output

    def save(self, directory_path: str):
        if not os.path.isdir(directory_path):
            os.mkdir(directory_path)

        with open(directory_path + "/model.pth", 'wb') as f:
            torch.save(self, f)

    def save_state(self, directory_path: str):
        if not os.path.isdir(directory_path):
            os.mkdir(directory_path)

        with open(directory_path + "/model_state.pth", 'wb') as f:
            torch.save(self.state_dict(), f)

    @staticmethod
    def load(directory_path: str):
        return torch.load(directory_path + '/model.pth', map_location=torch.device('cpu'))




def return_model(fname):
    with np.load(fname + '.npz', allow_pickle=True) as f:
        xprotos, yprotos = f['xprotos'], f['yprotos']
        lamda = f['lamda']
        print(f"train accuracy: {f['accuracy_of_train_set'][-1]}, "
              f"\t validation accuracy: {f['accuracy_of_validation_set'][-1]} ({np.max(f['accuracy_of_validation_set'])})")

    return xprotos, yprotos, lamda
