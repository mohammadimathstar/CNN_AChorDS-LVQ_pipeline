import yaml

from src.features.backbone_net import get_network
from src.AChorDSLVQ.model import Model
from src.train.optimizer_sgd import get_optimizer

from src.utils.logs import get_logger


def create_network(config_file_path, device):

    with open(config_file_path, "r") as file:
        config = yaml.safe_load(file)

    logger = get_logger('CREATE_NETWORK',
                        log_level=config['base']['log_level'],
                        log_file=config['base']['log_file'])

    features_net, add_on_layers = get_network(config)

    model = Model(config=config,
                  feature_extractor=features_net,
                  add_on_layers=add_on_layers,
                  device=device)

    model = model.to(device)
    
    logger.info(f"The network (with the backbone network '{config['feature']['backbone_net']}') has been created.")
    logger.info(f"The shape of prototypes is: {model.prototype_layer.xprotos.shape}")

    optimizer_net, optimizer_proto, optimizer_rel, params_to_freeze, params_to_train = get_optimizer(model, config_file_path)

    return model, optimizer_net, optimizer_proto, optimizer_rel, params_to_freeze, params_to_train

