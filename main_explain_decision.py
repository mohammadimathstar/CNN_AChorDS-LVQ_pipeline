import argparse
import os
import logging
from logging.handlers import RotatingFileHandler

from PIL import Image
import yaml

import torch
import torchvision.transforms as transforms

from src.AChorDSLVQ.model import Model
from src.explainability.feature_importance import (
    compute_feature_importance_heatmap, plot_important_region_per_principal_direction)
# from explain.args_explain import get_local_expl_args
from src.explainability.data_utils import load_and_process_images


# ------------------------------
# Logging Configuration
# ------------------------------
logger = logging.getLogger("ExplainAPI")
logger.setLevel(logging.INFO)

fomatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = RotatingFileHandler(
    filename='./explanations.log', maxBytes=10000, backupCount=1
)
file_handler.setFormatter(fomatter)
logger.addHandler(file_handler)


def explain_decision(config_file_path):
    """
    Explain model decisions by computing feature importance and visualizing regions of interest.
    """

    with open(config_file_path, "r") as file:
        config = yaml.safe_load(file)

    # Create results directory if it doesn't exist
    os.makedirs(config['explainability']['reports_dir'], exist_ok=True)

    # Update results directory path to include dataset name
    config['explainability']['reports_dir'] = os.path.join(config['explainability']['reports_dir'], 
                                                           config['data_loader']['data_name'])

    # Load the trained model
    model = Model.load(config['explainability']['model_path'])

    # Process images and obtain transformed data
    images_names, labels, transformed_images = load_and_process_images(config)

    # Compute feature importance heatmap
    region_importance_per_principal_dir, imgs = compute_feature_importance_heatmap(
        model, images_names, transformed_images, labels, config)

    # Plot and save important regions per principal direction
    plot_important_region_per_principal_direction(
        imgs, region_importance_per_principal_dir, images_names, config)


if __name__ == '__main__':
    # args = get_local_expl_args()
    explain_decision(config_file_path='params.yaml')