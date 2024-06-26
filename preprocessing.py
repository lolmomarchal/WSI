import subprocess
import numpy as np
import pandas as pd

OPENSLIDE_PATH = r"C:\Users\albao\Downloads\openslide-win64-20231011\openslide-win64-20231011\bin"
import Resources
import os

if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide
import histomicstk as htk
from PIL import Image
import cv2
import shutil
import matplotlib.pyplot as plt
import girder_client
from skimage.transform import resize
from matplotlib.colors import ListedColormap
from histomicstk.preprocessing.color_normalization import reinhard
from histomicstk.saliency.tissue_detection import (
    get_slide_thumbnail, get_tissue_mask)
from histomicstk.annotations_and_masks.annotation_and_mask_utils import (
    get_image_from_htk_response)
from histomicstk.preprocessing.color_normalization. \
    deconvolution_based_normalization import deconvolution_based_normalization
from histomicstk.preprocessing.color_deconvolution. \
    color_deconvolution import color_deconvolution_routine, stain_unmixing_routine
from histomicstk.preprocessing.augmentation. \
    color_augmentation import rgb_perturb_stain_concentration, perturb_stain_concentration
import gc
import time
import large_image
import torch
import csv
from torchvision import transforms
import torchstain
from pathlib import Path
from typing import TYPE_CHECKING
import matplotlib as mpl
import requests

if TYPE_CHECKING:
    import numpy as np

from tiatoolbox.tools.tissuemask import MorphologicalMasker
from tiatoolbox.wsicore.wsireader import WSIReader

"""
preprocessing.py

Author: Lorenzo Olmo Marchal
Created: March 12, 2024
Last Updated:  March 26, 2024

Description:
This script automates the preprocessing and normalization of Whole Slide Images (WSI) in digital pathology. 
It extracts tiles from WSI files and applies color normalization techniques to enhance image quality and consistency.

Input:
- slide directory path
- slide directory output path

Output:
Processed tiles are saved in the output directory. Each tile is accompanied by metadata, including its origin within the WSI file.

Future Directions:
- Integration of machine learning algorithms for automated tile selection and quality control
- Support for parallel processing on distributed computing platforms for handling large-scale WSI datasets efficiently
"""


def whole_slide_mask(slide_path, results):
    try:
        wsi = WSIReader.open(input_img=slide_path)
        wsi_thumb = wsi.slide_thumbnail(resolution=1, units="power")
        mask = wsi.tissue_mask(resolution=1, units="power")
        mask_thumb = mask.slide_thumbnail(
            resolution=1,
            units="power",
        )
        plt.imshow(mask_thumb)
        plt.axis("off")
        plt.savefig(os.path.join(results, "masked_slide"), bbox_inches='tight', pad_inches=0)
        plt.close()
        del mask_thumb
        gc.collect()

        plt.imshow(wsi_thumb)
        plt.axis("off")
        plt.savefig(os.path.join(results, "original_slide"), bbox_inches='tight', pad_inches=0)
        del wsi_thumb
        gc.collect()
    except Exception as e:
        print("file could not be opened", e)
        return None
    return mask


def is_background(org_image, image_mask, background_threshold=0.8, color_threshold=200):
    if org_image is None or image_mask is None:
        return True

    # Calculate the percentage of non-zero pixels in the mask
    percentage_mask_non_zero = np.sum(image_mask) / np.prod(image_mask.shape)

    # Calculate the percentage of white or light-colored pixels in the original image
    white_or_light = np.any(org_image > color_threshold, axis=-1)
    percentage_org_white_or_light = np.sum(white_or_light) / np.prod(org_image.shape[:-1])

    # Check if either percentage exceeds the threshold
    if percentage_mask_non_zero > background_threshold or percentage_org_white_or_light > background_threshold:
        return True

    return False


def tiling(slide, result_path, mask, overlap=False, stride=512):
    # Make tile dir
    os.makedirs(os.path.join(result_path, "tiles"), exist_ok=True)
    tiles = os.path.join(result_path, "tiles")
    columns = ['patient_id', 'x', 'y', 'magnification', 'path_to_slide']
    df_list = []

    ts = large_image.getTileSource(slide)
    size = 512
    if overlap != False:
        stride = stride
    else:
        stride = size
    mag = 20
    w = ts.getMetadata()['sizeX']
    h = ts.getMetadata()['sizeY']
    for x in range(0, w - size + 1, stride):
        for y in range(0, h - size + 1, stride):
            if x <= w - size and y <= h - size:
                tissue_rgb, _ = ts.getRegionAtAnotherScale(
                    sourceRegion=dict(left=x, top=y, width=size, height=size,
                                      units='base_pixels'),
                    targetScale=dict(magnification=mag),
                    format=large_image.tilesource.TILE_FORMAT_NUMPY)

                mask_region = mask.read_rect(location=(x, y), resolution=mag, units="power", size=(512, 512))
                if is_background(np.array(tissue_rgb), mask_region) == False:  # Check if the tile is not a background
                    tile = Image.fromarray(tissue_rgb)
                    tile.save(os.path.join(tiles, f"tile_w{x}_h{y}_mag{mag}.png"))
                    df_list.append({
                        'patient_id': "result1",
                        'x': x,
                        'y': y,
                        'magnification': mag,
                        'path_to_slide': os.path.join(tiles, f"tile_w{x}_h{y}_mag{mag}.png")
                    })
    df = pd.DataFrame(df_list, columns=columns)
    df.to_csv(os.path.join(result_path, "tile_information.csv"), index=False)
    del df
    gc.collect()


def normalize_tiles(tile_information, result_path):
    tiles = pd.read_csv(tile_information)
    path = os.path.join(result_path, "normalized_tiles")
    size = 512
    df_list = []
    columns = ['patient_id', 'x', 'y', 'magnification', 'path_to_slide']
    reference_image_path = "./Resources/target.png"
    size = 512
    target = cv2.resize(cv2.cvtColor(cv2.imread(reference_image_path), cv2.COLOR_BGR2RGB), (size, size))
    for i, row in tiles.iterrows():
        tile = row["path_to_slide"]
        mag = row["magnification"]
        y = row["y"]
        x = row["x"]
        os.makedirs(os.path.join(result_path, "normalized_tiles"), exist_ok=True)
        tile = cv2.resize(cv2.cvtColor(cv2.imread(tile), cv2.COLOR_BGR2RGB), (size, size))
        try:
            if tile is not None:
                target = cv2.resize(cv2.cvtColor(cv2.imread(reference_image_path), cv2.COLOR_BGR2RGB), (size, size))
                T = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x * 255)
                ])

                torch_normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
                torch_normalizer.fit(T(target))
                t_to_transform = T(tile)

                norm, H, E = torch_normalizer.normalize(I=t_to_transform, stains=True)

                rgb_tensor = torch.tensor(norm, dtype=torch.float32) / 255.0
                rgb_tensor = rgb_tensor.permute(2, 0, 1)
                pil_image = transforms.ToPILImage()(rgb_tensor)
                pil_image.save(os.path.join(path, f"tile_w{x}_h{y}_mag{mag}.png"))
                df_list.append({
                    'patient_id': "result1",
                    'x': x,
                    'y': y,
                    'magnification': mag,
                    'path_to_slide': os.path.join(path, f"tile_w{x}_h{y}_mag{mag}.png")
                })

            else:
                print("Error: Input tile is None.")

        except Exception as e:
            print(f"An error occurred: {e}")
    df = pd.DataFrame(df_list, columns=columns)
    df.to_csv(os.path.join(result_path, "normalized_tile_information.csv"), index=False)
    del df
    gc.collect()


def call_build_script():
    try:
        subprocess.run(["python", "build.py"], check=True)
        print("build.py script executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing build.py script: {e}")


def move_svs_files(main_directory, results_path):
    # Create a CSV file to store patient ID and SVS file paths

    # Iterate through each patient directory
    for patient_directory in os.listdir(main_directory):
        patient_path = os.path.join(main_directory, patient_directory)

        if os.path.isdir(patient_path):
            # Look for SVS files in the patient directory
            for file_name in os.listdir(patient_path):
                if file_name.endswith(".svs"):
                    svs_file_path = os.path.join(patient_path, file_name)

                    # Move the SVS file to the main directory
                    shutil.move(svs_file_path, main_directory)


def patient_csv(main_directory, results_path):
    csv_file_path = os.path.join(results_path, "patient_files.csv")

    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Patient ID", "Original Slide Path", "Preprocessing Path"])
        for file in os.listdir(main_directory):

            if file.endswith(".svs"):

                patient_id = file.split("-")[2] + "-" + file.split("-")[3] + "-" + file.split("-")[4] + "-" + \
                             file.split("-")[5]
                patient_id = patient_id.split(".")[0]
                patient_result_path = os.path.join(results_path, patient_id)

                if not os.path.exists(patient_result_path):
                    os.makedirs(patient_result_path)
                csv_writer.writerow([patient_id, os.path.join(main_directory, file),
                                     os.path.join(results_path, patient_id)])
    return csv_file_path


def preprocessing(path, result_path):
    mask = whole_slide_mask(path, result_path)
    print(f"processing: {path}")
    if mask is not None:
        tile_inf_path = os.path.join(result_path, "tile_information.csv")
        if not os.path.isfile(tile_inf_path):
            tiling(path, result_path, mask)
        normalize_tiles_path = os.path.join(result_path, "normalized_tile_information.csv")
        if not os.path.isfile(normalize_tiles_path):
            normalize_tiles(tile_inf_path, result_path)


def main():
    call_build_script()

    path = input("Enter input path (or leave blank for default): ").strip()
    if not path:
        path = r"C:\Users\albao\Downloads\gdc_download_20240320_111546.230274"  # Provide default input path here
    result_path = input("Enter result path (or leave blank for default): ").strip()
    if not result_path:
        result_path = r"C:\Users\albao\Masters\WSI_results"  # Provide default result path here

    # for TCGA project patient ID:
    #  ex: "TCGA-BH-A0BC-01A-02-TSB.a7afb5f2-2676-428b-95c4-5b8764004820.svs"  would be "A0BC"
    # Perform preprocessing

    move_svs_files(path, result_path)
    patient_path = patient_csv(path, result_path)

    patients = pd.read_csv(patient_path)
    for i, row in patients.iterrows():
        result = row["Preprocessing Path"]
        original = row["Original Slide Path"]
        patient_id = row["Patient ID"]
        preprocessing(original, result)
        print(f"done with patient {patient_id}")


if __name__ == "__main__":
    main()
