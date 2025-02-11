# import logging

import torch
import torchvision.transforms as transforms

import yaml
import os
from PIL import Image
import pickle

from src.utils.logs import get_logger


def load_class_mapping(dataset_name: str) -> dict:
    """
    Load the class index to label mapping from a pickle file.
    """
    mapping_file = os.path.join('./data', f'index2label_{dataset_name}.pkl')
    with open(mapping_file, 'rb') as file:
        class_mapping = pickle.load(file)
    return class_mapping


def create_image_transform(image_size: int) -> transforms.Compose:
    """
    Create a preprocessing pipeline for image transformation.
    """
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean, std=std)

    transform_pipeline = transforms.Compose([
        transforms.Resize(size=(image_size, image_size)),
        transforms.ToTensor(),
        normalize
    ])

    return transform_pipeline


def extract_class_from_filename(filename: str, data_name: str) -> str:
    """
    Extract the class name from the image filename based on the dataset type.
    """
    if data_name == 'CARS':
        return filename.split("_")[0]
    elif data_name == 'CUB-200-2011':
        return "_".join(filename.split("_")[:-2])
    elif data_name == 'BRAIN':
        dic = {'gl': 'glioma', 'me': 'meningioma', 'no': 'notumor', 'pi': 'pituitary'}
        return dic[filename[3:5]]
    elif data_name == 'PETS':
        return "_".join(filename.split("_")[:-1])
    else:
        raise ValueError(f"Invalid dataset name provided: {data_name}")


def process_images(transform: transforms.Compose,
                   class_mapping: dict,
                   config):
    """
    Process images by applying transformations and saving results.
    """

    logger = get_logger('LOAD_IMAGES',
                        log_level=config['base']['log_level'],
                        log_file=config['base']['log_file_explainability'])
    
    images_dir = config['explainability']['sample_dir']
    label_to_index = {label: idx for idx, label in class_mapping.items()}

    processed_images = []
    image_filenames = []
    image_labels = []

    # Iterate through all images in the dataset directory
    for filename in os.listdir(images_dir):
        # Extract class name from the image filename
        class_name = extract_class_from_filename(filename, config['data_loader']['data_name'])

        # Get the corresponding class index
        image_labels.append(label_to_index[class_name])

        # Load and transform the image
        image_path = os.path.join(images_dir, filename)
        image = Image.open(image_path)
        processed_images.append(transform(image))
        image_filenames.append(filename)

        # Save the processed image in the appropriate results folder
        image_basename = os.path.splitext(filename)[0]
        destination_folder = os.path.join(config['explainability']['reports_dir'], image_basename)
        os.makedirs(destination_folder, exist_ok=True)

        image.save(os.path.join(destination_folder, filename))

    # Stack images into a single tensor
    processed_images = torch.stack(processed_images, dim=0)
    image_labels = torch.tensor(image_labels, dtype=torch.int64)

    logger.info(f"Processed {processed_images.shape[0]} images and saved results to '{config['explainability']['reports_dir']}'.\n")

    return image_filenames, image_labels, processed_images


def load_and_process_images(config: dict):
    """
    Load class mappings, create preprocessing pipeline, and process images.
    """

    class_mapping = load_class_mapping(config['data_loader']['data_name'])
    transform_pipeline = create_image_transform(config['data_loader']['image_size'])

    return process_images(transform_pipeline, class_mapping, config)


