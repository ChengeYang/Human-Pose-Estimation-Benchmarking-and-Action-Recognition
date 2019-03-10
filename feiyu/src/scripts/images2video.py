

import numpy as np
import cv2
import sys, os, time
CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"

image_folder = CURR_PATH + '../skeleton_data/skeletons3_images/'
video_name = CURR_PATH + '../skeleton_data/skeletons3.avi'



# Settings
# image_start = 770
# image_end = 1560
image_start = 922
image_end = 1180
framerate = 7
width = 640
height = 480
#images_names = [img for img in os.listdir(image_folder) if img.endswith(".png")] # need to sort this

# Read image and save to video'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter(video_name, fourcc, framerate, (width,height))
cnt = 0
for i in range(image_start, image_end+1):
    cnt += 1
    fname = "{:05d}.png".format(i)
    print("Processing the {}/{}th image: {}".format(cnt, image_end - image_start + 1, fname))
    video.write(cv2.imread(os.path.join(image_folder, fname)))

cv2.destroyAllWindows()
video.release()