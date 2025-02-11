
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.explainability.importance_scores import compute_feature_importance, save_feature_importance_heatmap
import torch

from typing import List

from src.AChorDSLVQ.model import Model
from src.utils.logs import get_logger


def k_largest_index_argsort(a, k):
    idx = np.argsort(a.ravel())[:-k-1:-1]
    return np.column_stack(np.unravel_index(idx, a.shape))


def plot_important_region_per_principal_direction(imgs,
        region_importances_per_principal_direction,
        imgs_names,
        config: dict):
    
    k = 1
    
    regionW, regionH = region_importances_per_principal_direction.shape[-2:]

    region_size = (int(config['data_loader']['image_size']/regionW), int(config['data_loader']['image_size']/regionH))

    for img_name, img, region_map in zip(imgs_names, imgs, region_importances_per_principal_direction):
        result_dir = os.path.join(config['explainability']['reports_dir'], img_name[:-4], 'regions_per_directions')
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        for i, region_per_dir in enumerate(region_map):
            region_per_dir = region_per_dir.numpy().T # TODO: Check for transpose
            # region_per_dir = np.rot90(region_per_dir.numpy(), 2) #k=2,

            image = cv2.resize(img, (config['data_loader']['image_size'], config['data_loader']['image_size']), interpolation=cv2.INTER_LINEAR)
            ids = k_largest_index_argsort(region_per_dir, k=k)

            for idx in ids:

                row_idx, col_idx = idx[0], idx[1]

                # row_idx = regionH - row_idx -1 # %TODO: check (check line 40 above)
                # col_idx = regionW - col_idx - 1 # %TODO: check (check line 40 above)

                # Start coordinate: represents the top left corner of rectangle
                start_point = (row_idx * region_size[0], col_idx * region_size[1])

                # Ending coordinate: represents the bottom right corner of rectangle
                end_point = (start_point[0] + region_size[0], start_point[1] + region_size[1])

                # Blue color in BGR
                color = (255, 0, 0)

                # Line thickness of 2 px
                thickness = 2

                # Draw a rectangle with blue line borders of thickness of 2 px
                image = cv2.rectangle(image, start_point, end_point, color, thickness)

                plt.imsave(
                        fname=os.path.join(result_dir, '%i.png' % i),
                        arr=image,
                        vmin=0.0, vmax=1.0
                    )


def compute_feature_importance_heatmap(model: Model,
                 img_names: List,
                 imgs_transformed: torch.Tensor,
                 labels,
                #  logger,
                 config: dict):
    
    logger = get_logger('COMPUTE_FEATURE_IMPORTANCE_HEATMAP',
                        log_level=config['base']['log_level'],
                        log_file=config['base']['log_file'])

    OUTPUT_DIR = config['explainability']['reports_dir']

    region_effect_maps = []
    region_effect_maps_per_principal_direction = []
    images_resized = []
    for img_name, label, sample in zip(img_names, labels, imgs_transformed):

        # fname = ".".join(img_name.split(".")[:-1])
        fname = os.path.splitext(img_name)[0]
        print("\n", fname)

        INPUT_PATH = os.path.join(OUTPUT_DIR, fname, img_name)
        HEATMAP_PATH = os.path.join(OUTPUT_DIR, fname, 'heatmap.png')

        image = cv2.imread(INPUT_PATH)

        image_resized = cv2.resize(image, (config['data_loader']['image_size'], config['data_loader']['image_size']),
                                   interpolation=cv2.INTER_LINEAR)

        images_resized.append(image)

        with (torch.no_grad()):
            feature, subspace, Vh, S, output = model.forward_partial(sample.unsqueeze(0))

            region_heatmap, region_heatmap_per_principal_dir = compute_feature_importance(
                feature, label, Vh, S, output,
                model.prototype_layer.xprotos,
                model.prototype_layer.yprotos_mat,
                model.prototype_layer.yprotos_comp_mat,
                model.prototype_layer.relevances,
                return_full_output=False
            )

        save_feature_importance_heatmap(region_heatmap, output_path=HEATMAP_PATH)
        logger.info(f"The importance of regions (of '{img_name}') has been completed!")
        logger.info(f"Its heatmap has been saved in '{HEATMAP_PATH}'.")

        # Resize to image size and save the (upsampled) heatmap
        heatmap_upsampled = cv2.resize(
            region_heatmap.numpy(),
            dsize=(config['data_loader']['image_size'], config['data_loader']['image_size']), #(sample_array.shape[1], sample_array.shape[0]),
            interpolation=cv2.INTER_CUBIC
        )

        UPSAMPLED_HEATMAP_PATH = os.path.join(OUTPUT_DIR, fname, 'heatmap_upsampled.png')
        heatmap_upsampled_normalized = save_feature_importance_heatmap(heatmap_upsampled, UPSAMPLED_HEATMAP_PATH)

        overlay = 0.5 * image_resized / 255 + 0.3 * heatmap_upsampled_normalized
        OVERLAY_PATH = os.path.join(OUTPUT_DIR, fname, 'heatmap_original_image.png')
        plt.imsave(fname=OVERLAY_PATH, arr=overlay, vmin=0.0, vmax=1.0)

        logger.info(f"The image with its heatmap has been saved in '{OVERLAY_PATH}'.\n")

        region_effect_maps_per_principal_direction.append(
            region_heatmap_per_principal_dir
        )

    return (
        torch.stack(region_effect_maps_per_principal_direction, dim=0),
        images_resized
    )


