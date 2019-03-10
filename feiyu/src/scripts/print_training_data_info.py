

import numpy as np
import sys, os, time
from mylib.io import *
CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"



data_idx = "2"
SRC_IMAGE_FOLDER = CURR_PATH + "../../data/source_images"+data_idx+"/"
VALID_IMAGES_TXT = "valid_images.txt"

# Run this function, and it will print out the info of training images
images_info = collect_images_info_from_source_images(
    SRC_IMAGE_FOLDER, VALID_IMAGES_TXT)
