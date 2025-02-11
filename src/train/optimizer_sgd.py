import yaml

import torch.optim


def get_optimizer(model, config_file_path) -> tuple:
    """
    Construct the optimizer as dictated by the parsed arguments
    :param model: The model that should be optimized
    :param config_file_path: Path to the YAML configuration file
    :return: the optimizer corresponding to the parsed arguments, parameter set that can be frozen,
    and parameter set of the net that will be trained
    """

    with open(config_file_path, "r") as file:
        config = yaml.safe_load(file)

    backbone_net = config['feature']['backbone_net']
    momentum = config['train']['momentum']
    weight_decay = config['train']['weight_decay']
    
    #create parameter groups
    params_to_freeze = []
    params_to_train = []

    # set up optimizer
    if ('resnet50_inat' in backbone_net) or ('convnext_tiny_13' in backbone_net) or ('resnet50' in backbone_net):  
        # freeze resnet50 except last convolutional layer
        for name, param in model.feature_extractor.named_parameters():
            if 'layer4.2' not in name:
                params_to_freeze.append(param)
            else:
                params_to_train.append(param)

        net_paramlist = [
            {"params": params_to_freeze, "lr": config['train']['lr_net'], "weight_decay_rate": weight_decay, "momentum": momentum},
            {"params": params_to_train, "lr": config['train']['lr_block'], "weight_decay_rate": weight_decay,"momentum": momentum},
            # {"params": model._add_on.parameters(), "lr": args.lr_block, "weight_decay_rate": args.weight_decay,"momentum": args.momentum},
        ]
        proto_paramlist = [
            {"params": model.prototype_layer.xprotos,
             "lr": config['train']['lr_prototypes'],
             "weight_decay_rate": 0,
             "momentum": 0
             },
        ]
        rel_paramlist = [
            {"params": model.prototype_layer.relevances, "lr": config['train']['lr_lambda'], "weight_decay_rate": weight_decay, "momentum": momentum}
        ]
    
    # else: #other network architectures
    #     for name, param in model._net.named_parameters():
    #         params_to_freeze.append(param)
    #     paramlist = [
    #         {"params": params_to_freeze, "lr": args.lr_net, "weight_decay_rate": args.weight_decay},
    #         # {"params": model._add_on.parameters(), "lr": args.lr_block}, #"weight_decay_rate": args.weight_decay},
    #         {"params": model.prototype_layer.parameters(), "lr": args.lr_protos, "weight_decay_rate": 0},
    #         {"params": model.relevances.parameters(), "lr": args.lr_rel, "weight_decay_rate": 0}
    #     ]


    return torch.optim.Adam(net_paramlist), torch.optim.SGD(proto_paramlist), torch.optim.Adam(rel_paramlist), params_to_freeze, params_to_train
