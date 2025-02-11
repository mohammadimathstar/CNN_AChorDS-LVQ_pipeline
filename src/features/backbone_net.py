import yaml
import torch.nn as nn
import torch.nn.functional as F
# from prototree.prototree import ProtoTree
from torch.nn import Module
# from util.log import Log
from src.features.resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet50_features_inat, resnet101_features, resnet152_features
from src.features.densenet_features import densenet121_features, densenet161_features, densenet169_features, densenet201_features
from src.features.vgg_features import vgg11_features, vgg11_bn_features, vgg13_features, vgg13_bn_features, vgg16_features, vgg16_bn_features,vgg19_features, vgg19_bn_features
from src.features.convnext import convnext_tiny
from src.features.convnext_features import convnext_tiny_13_features, convnext_tiny_26_features
from src.utils.logs import get_logger

base_architecture_to_features = {'resnet18': resnet18_features,
                                 'resnet34': resnet34_features,
                                 'resnet50': resnet50_features,
                                 'resnet50_inat': resnet50_features_inat,
                                 'resnet101': resnet101_features,
                                 'resnet152': resnet152_features,
                                 'densenet121': densenet121_features,
                                 'densenet161': densenet161_features,
                                 'densenet169': densenet169_features,
                                 'densenet201': densenet201_features,
                                 'vgg11': vgg11_features,
                                 'vgg11_bn': vgg11_bn_features,
                                 'vgg13': vgg13_features,
                                 'vgg13_bn': vgg13_bn_features,
                                 'vgg16': vgg16_features,
                                 'vgg16_bn': vgg16_bn_features,
                                 'vgg19': vgg19_features,
                                 'vgg19_bn': vgg19_bn_features,
                                 # 'convnext_tiny_1k': convnext_tiny,
                                 # 'convnext_tiny_26': convnext_tiny_26_features,
                                 'convnext_tiny_13': convnext_tiny_13_features
                                 }

"""
    Create network with pretrained features and 1x1 convolutional layer

"""
def get_network(config):


    # Define a conv net for estimating the probabilities at each decision node
    # features = base_architecture_to_features[args.net](pretrained=not args.disable_pretrained) 
    features = base_architecture_to_features[config['feature']['backbone_net']](pretrained=True)           
    features_name = str(features).upper()
    if features_name.startswith('VGG') or features_name.startswith('RES') or features_name.startswith('CONVNEXT'):
        first_add_on_layer_in_channels = \
            [i for i in features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
    elif features_name.startswith('DENSE'):
        first_add_on_layer_in_channels = \
            [i for i in features.modules() if isinstance(i, nn.BatchNorm2d)][-1].num_features
    else:
        raise Exception('other base base_architecture NOT implemented')
    

    add_on_layers = nn.Sequential(
        nn.Conv2d(in_channels=first_add_on_layer_in_channels, 
                  out_channels=config['train']['hyperparams']['prototype_depth'],#args.num_features, 
                  kernel_size=1,
                  bias=False),
        nn.Sigmoid(), # for prototree
        #nn.Softmax(dim=1), #for pipnet: softmax over every prototype for each patch, such that for every location in image, sum over prototypes is 1      
    )
    return features, add_on_layers

def freeze(net: Module,
           epoch: int,
           params_to_freeze: list,
        #    log: Log,
           config: dict):
    """
    Freeze the network for a specified number of epochs during training.

    Parameters
    ----------
    epoch : int
        The current epoch number.
    params_to_freeze : list
        A list of parameters to freeze.
    log : Log
        An instance of the :class:`Log` class to log messages.
    config : dict
        The dictionary of configuration parameters.

    Returns
    -------
    None
    """

    logger = get_logger('FREEZE',
                        log_level=config['base']['log_level'],
                        log_file=config['base']['log_file'])
    
    if config['train']['freeze_epochs'] > 0:
        if epoch == 1:
            logger.info("Network frozen")
            for parameter in params_to_freeze:
                parameter.requires_grad = False
        elif epoch == config['train']['freeze_epochs'] + 1:
            logger.info("Network unfrozen")
            for parameter in params_to_freeze:
                parameter.requires_grad = True

