import yaml
import torch

from src.data.preprocessing_data import get_data
from src.utils.logs import get_logger


def get_dataloaders(config_file_path: str):
    """
    Prepare data loaders for training and testing.

    :param config_file_path: the path to the YAML configuration file. 
    :return: Tuple containing trainloader, testloader, class names, and input shape.
    """

    with open(config_file_path, "r") as file:
        config = yaml.safe_load(file)

    logger = get_logger('DATA_LOADERS',
                        log_level=config['base']['log_level'],
                        log_file=config['base']['log_file'])

    trainset, testset, classes, shape = get_data(config)
    cuda_available = config["base"]["use_cuda"] and torch.cuda.is_available()

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=config['data_loader']["batch_size_train"], shuffle=True, pin_memory=cuda_available
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=config['data_loader']["batch_size_test"], shuffle=False, pin_memory=cuda_available
    )

    logger.info(f"Data loaders have been created.")

    return trainloader, testloader, classes, shape