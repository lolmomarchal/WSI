# imports.py

import numpy as np
import pandas as pd
import os
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from skimage.transform import resize
from matplotlib.colors import ListedColormap
import gc
import time
import large_image
import torch
from torchvision import transforms
import torchstain
import matplotlib as mpl
import requests
from typing import TYPE_CHECKING
import csv

if TYPE_CHECKING:
    import numpy as np
from tiatoolbox.tools.tissuemask import MorphologicalMasker
from tiatoolbox.wsicore.wsireader import WSIReader

# Check if OS is Windows
if os.name == 'nt':
    # Ask user for OpenSlide path if not specified
    OPENSLIDE_PATH = input("Enter the path to OpenSlide (leave empty for default): ")
    if not OPENSLIDE_PATH:
        # Set default path
        OPENSLIDE_PATH = r"C:\Users\albao\Downloads\openslide-win64-20231011\openslide-win64-20231011\bin"

    # Handle OpenSlide import based on path
    if hasattr(os, 'add_dll_directory'):
        with os.add_dll_directory(OPENSLIDE_PATH):
            import openslide
    else:
        import openslide
else:
    # For non-Windows platforms, just import openslide normally
    import openslide

# HistomicsTK import handling
import histomicstk as htk
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

# Girder Client import
import girder_client

