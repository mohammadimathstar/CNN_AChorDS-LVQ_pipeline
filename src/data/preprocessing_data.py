import os
import yaml
import pickle
from typing import Tuple, List

import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import Compose, ToTensor, Normalize


def load_params(file_path: str = "params.yaml") -> dict:
    """
    Load configuration parameters from a YAML file.

    :param file_path: Path to the YAML file (default: "params.yaml").
    :return: Dictionary of parameters.
    """
    with open(file_path, "r") as file:
        params = yaml.safe_load(file)
    return params


def get_dataset_birds(params: dict):
    """
    Load and preprocess a dataset with optional augmentation.

    :param params: Configuration dictionary containing the dataset name and paths.
    :return: Tuple containing trainset, testset, class names, and input shape.
    """
    shape = (3, params['data_loader']['image_size'], params['data_loader']['image_size'])
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    normalize = Normalize(mean=mean, std=std)
    transform_basic = Compose([
        transforms.Resize(size=(params['data_loader']['image_size'], params['data_loader']['image_size'])),
        ToTensor(),
        normalize
    ])

    
    transform_augment = Compose([
        transforms.Resize(size=(params['data_loader']['image_size'], params['data_loader']['image_size'])),  
        transforms.RandomOrder([
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4), hue=(-0.02, 0.02)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=10, shear=(-2, 2), translate=[0.05,0.05]),
        ]),
        ToTensor(),
        normalize
    ])

    trainset = torchvision.datasets.ImageFolder(params['data_loader']['train_dir'], transform=transform_augment)
    testset = torchvision.datasets.ImageFolder(params['data_loader']['test_dir'], transform=transform_basic)
    classes = trainset.classes

    # Special handling for certain datasets
    if params['data_loader']['data_name'] == "CUB-200-2011":
        classes = [cls.split('.')[1] for cls in classes]  # Remove numerical prefixes

    return trainset, testset, classes, shape


def get_dataset_cars(params: dict):
    """
    Load and preprocess a dataset with optional augmentation.

    :param params: Configuration dictionary containing the dataset name and paths.
    :return: Tuple containing trainset, testset, class names, and input shape.
    """
    shape = (3, params['data_loader']['image_size'], params['data_loader']['image_size'])
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    normalize = Normalize(mean=mean, std=std)
    transform_basic = Compose([
        transforms.Resize(size=(params['data_loader']['image_size'], params['data_loader']['image_size'])),
        ToTensor(),
        normalize
    ])

    
    transform_augment = Compose([
        transforms.Resize(size=(params['data_loader']['image_size'] + 32, params['data_loader']['image_size'] + 32)),  # Resize slightly larger
        transforms.RandomOrder([
            transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
            transforms.ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4), hue=(-0.4, 0.4)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=15, shear=(-2, 2)),
        ]),
        transforms.RandomCrop(size=(params['data_loader']['image_size'], params['data_loader']['image_size'])),
        ToTensor(),
        normalize
    ])

    trainset = torchvision.datasets.ImageFolder(params['data_loader']['train_dir'], transform=transform_augment)
    testset = torchvision.datasets.ImageFolder(params['data_loader']['test_dir'], transform=transform_basic)
    classes = trainset.classes

    return trainset, testset, classes, shape


def get_data(params: dict):
    """
    Load dataset based on configuration parameters.

    :param params: Configuration dictionary containing the dataset name and paths.
    :return: Tuple containing trainset, testset, classes, and input shape.
    """
    datasets_config = {
        "CUB-200-2011": ("./data/CUB_200_2011/dataset/train_corners",
                         "./data/CUB_200_2011/dataset/test_full"),
        "CARS": ("./data/cars/dataset/train",
                 "./data/cars/dataset/test"),
        "PETS": ("./data/PETS/dataset/train",
                 "./data/PETS/dataset/test"),
        "BRAIN": ("./data/brain-tumor/Training",
                  "./data/brain-tumor/Testing"),
        "MURA": ("./data/MURA-v1.1/dataset/train",
                 "./data/MURA-v1.1/dataset/valid"),
        "ETH-80": ("./data/ETH-80/train",
                   "./data/ETH-80/test")
    }

    dataset_name = params['data_loader']["data_name"]
    if dataset_name not in datasets_config:
        raise ValueError(f'Unknown dataset: "{dataset_name}"')

    if dataset_name in ['CUB-200-2011', 'PETS']:
        return get_dataset_birds(params)
    else:
        return get_dataset_cars(params)





# if __name__ == "__main__":
#     # Load configuration parameters
#     params = load_params()

#     # Prepare data loaders and other components
#     trainloader, testloader, classes, shape = get_dataloaders(params)

#     # Generate class-to-index mapping
#     index_to_label = {i: label for i, label in enumerate(classes)}
#     print(index_to_label)

#     # Save the mapping as a pickle file
#     output_path = f'index2label_{params["dataset"]}.pkl'
#     with open(output_path, 'wb') as f:
#         pickle.dump(index_to_label, f)
#     print(f"Class index mapping saved to {output_path}")
