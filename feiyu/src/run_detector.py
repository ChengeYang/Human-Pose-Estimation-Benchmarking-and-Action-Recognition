
'''
This script runs the method from the github repo of "tf-pose-estimation"
https://github.com/ildoonet/tf-pose-estimation

press 'q' to quit
press others to continue testing other images

'''

import numpy as np
import cv2
import sys, os, time, argparse, logging
import simplejson
import argparse

import mylib.io as myio
import mylib.funcs as myfunc
from mylib.feature_proc import ProcFtr 
from mylib.action_classifier import *

# PATHS ==============================================================

CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"


# INPUTS ==============================================================

def parse_input_FROM_WEBCAM():
    key_word = "--images_source"
    choices = ["webcam", "folder"]

    parser = argparse.ArgumentParser()
    parser.add_argument(key_word, required=False, default='webcam')
    inp = parser.parse_args().images_source
    if inp == "webcam":
        return True
    elif inp == "folder":
        return False
    else:
        print("\nWrong command line input !\n")
        assert True


# PATHS and SETTINGS =================================

data_idx = "2"
SRC_IMAGE_FOLDER = CURR_PATH + "../data/source_images"+data_idx+"/"
VALID_IMAGES_TXT = "valid_images.txt"

SKELETON_FOLDER = "skeleton_data/"
SAVE_DETECTED_SKELETON_TO =         "skeleton_data/skeletons"+data_idx+"/"
SAVE_DETECTED_SKELETON_IMAGES_TO =  "skeleton_data/skeletons"+data_idx+"_images/"
SAVE_IMAGES_INFO_TO =               "skeleton_data/images_info"+data_idx+".txt"

FROM_WEBCAM = parse_input_FROM_WEBCAM()

DO_INFERENCE =  True and FROM_WEBCAM
DO_INFERENCE_MODEL = ["EECS_433", "EECS_496"][1]
SAVE_RESULTANT_SKELETON_TO_TXT_AND_IMAGE = not FROM_WEBCAM
# SAVE_RESULTANT_SKELETON_TO_TXT_AND_IMAGE = False

# create folders ==============================================================
if not os.path.exists(CURR_PATH+SKELETON_FOLDER):
    os.makedirs(CURR_PATH+SKELETON_FOLDER)
if not os.path.exists(CURR_PATH+SAVE_DETECTED_SKELETON_TO):
    os.makedirs(CURR_PATH+SAVE_DETECTED_SKELETON_TO)
if not os.path.exists(CURR_PATH+SAVE_DETECTED_SKELETON_IMAGES_TO):
    os.makedirs(CURR_PATH+SAVE_DETECTED_SKELETON_IMAGES_TO)

# Openpose ==============================================================

sys.path.append(CURR_PATH + "githubs/tf-pose-estimation")
from tf_pose.networks import get_graph_path, model_wh
from tf_pose.estimator import TfPoseEstimator
from tf_pose import common

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


# Human pose detection ==============================================================

class SkeletonDetector(object):
    # This func is copied from https://github.com/ildoonet/tf-pose-estimation

    def __init__(self, model="cmu"):
        models = set({"mobilenet_thin", "cmu"})
        self.model = model if model in models else "mobilenet_thin"
        # parser = argparse.ArgumentParser(description='tf-pose-estimation run')
        # parser.add_argument('--image', type=str, default='./images/p1.jpg')
        # parser.add_argument('--model', type=str, default='cmu', help='cmu / mobilenet_thin')

        # parser.add_argument('--resize', type=str, default='0x0',
        #                     help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
        # parser.add_argument('--resize-out-ratio', type=float, default=4.0,
        #                     help='if provided, resize heatmaps before they are post-processed. default=1.0')
        self.resize_out_ratio = 4.0

        # args = parser.parse_args()

        # w, h = model_wh(args.resize)
        w, h = model_wh("432x368")
        if w == 0 or h == 0:
            e = TfPoseEstimator(get_graph_path(self.model),
                                target_size=(432, 368))
        else:
            e = TfPoseEstimator(get_graph_path(self.model), target_size=(w, h))

        # self.args = args
        self.w, self.h = w, h
        self.e = e
        self.fps_time = time.time()

    def detect(self, image):
        t = time.time()

        # Inference
        humans = self.e.inference(image, resize_to_default=(self.w > 0 and self.h > 0),
                                #   upsample_size=self.args.resize_out_ratio)
                                  upsample_size=self.resize_out_ratio)

        # Print result and time cost
        print("humans:", humans)
        elapsed = time.time() - t
        logger.info('inference image in %.4f seconds.' % (elapsed))

        return humans
    
    def draw(self, img_disp, humans):
        img_disp = TfPoseEstimator.draw_humans(img_disp, humans, imgcopy=False)

        logger.debug('show+')
        cv2.putText(img_disp,
                    "FPS: %f" % (1.0 / (time.time() - self.fps_time)),
                    (20, 20),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)
        self.fps_time = time.time()

    @staticmethod
    def humans_to_skelsInfo(humans, action_type="None"):
        # skeleton = [action_type, 18*[x,y], 18*score]
        skelsInfo = []
        NaN = 0
        for human in humans:
            skeleton = [NaN]*(1+18*2+18)
            skeleton[0] = action_type
            for i, body_part in human.body_parts.items(): # iterate dict
                idx = body_part.part_idx
                skeleton[1+2*idx]=body_part.x
                skeleton[1+2*idx+1]=body_part.y
                # skeleton[1+36+idx]=body_part.score
            skelsInfo.append(skeleton)
        return skelsInfo
    
    @staticmethod
    def get_ith_skeleton(skelsInfo, ith_skeleton=0):
        return np.array(skelsInfo[ith_skeleton][1:1+18*2])


# ==============================================================



class DataLoader_usbcam(object):
    def __init__(self):
        self.cam = cv2.VideoCapture(0)
        self.num_images = 9999999

    def load_next_image(self):
        ret_val, img = self.cam.read()
        img =cv2.flip(img, 1)
        action_type = "unknown"
        return img, action_type

class DataLoader_imagesfolder(object):
    def __init__(self, SRC_IMAGE_FOLDER, VALID_IMAGES_TXT):
        self.images_info = myio.collect_images_info_from_source_images(SRC_IMAGE_FOLDER, VALID_IMAGES_TXT)
        self.imgs_path = SRC_IMAGE_FOLDER
        self.i = 0
        self.num_images = len(self.images_info)
        print("Reading images from folder: {}\n".format(SRC_IMAGE_FOLDER))
        print("Reading images information from: {}\n".format(VALID_IMAGES_TXT))
        print("    Num images = {}\n".format(self.num_images))

    def save_images_info(self, path):
        with open(path, 'w') as f:
            simplejson.dump(self.images_info, f)

    def load_next_image(self):
        self.i += 1
        filename = self.get_filename(self.i)
        img = self.imread(self.i)
        action_type = self.get_action_type(self.i)
        return img, action_type

    def imread(self, index):
        return cv2.imread(self.imgs_path + self.get_filename(index))
    
    def get_filename(self, index):
        # [1, 7, 54, "jump", "jump_03-02-12-34-01-795/00240.png"]
        # See "myio.collect_images_info_from_source_images" for the data format
        return self.images_info[index-1][4] 

    def get_action_type(self, index):
        # [1, 7, 54, "jump", "jump_03-02-12-34-01-795/00240.png"]
        # See "myio.collect_images_info_from_source_images" for the data format
        return self.images_info[index-1][3]
        

# ==============================================================
class ActionClassifier(object):
    
    def __init__(self, model_path):
        from keras.models import load_model

        self.dnn_model = load_model(model_path)
        self.action_dict = ["kick", "punch", "squat", "stand", "wave"]

    def predict(self, skeleton):

        # Preprocess data
        if 0:
            skeleton_input = ProcFtr.pos2angles(skeleton).reshape(-1, 8)
        else:
            tmp = ProcFtr.pose_normalization(skeleton)
            skeleton_input = np.array(tmp).reshape(-1, len(tmp))
            
        # Predicted label: int & string
        predicted_idx = np.argmax(self.dnn_model.predict(skeleton_input))
        prediced_label = self.action_dict[predicted_idx]

        return prediced_label


if __name__ == "__main__":
 
    # -- Detect sekelton
    my_detector = SkeletonDetector()

    # -- Load images
    if FROM_WEBCAM:
        images_loader = DataLoader_usbcam()

    else:
        images_loader = DataLoader_imagesfolder(SRC_IMAGE_FOLDER, VALID_IMAGES_TXT)
        images_loader.save_images_info(path = CURR_PATH + SAVE_IMAGES_INFO_TO)

    # -- Classify action
    if DO_INFERENCE:
        if DO_INFERENCE_MODEL == "EECS_496":
            classifier = ActionClassifier(
                CURR_PATH + "githubs/Skeleton-Based-Human-Action-Recognition/action_recognition.h5"
            )
        else:
            classifier = MyClassifier(
                CURR_PATH + "trained_classifier.pickle",
                # action_types = ['jump', 'kick', 'run', 'sit', 'squat', 'stand', 'walk', 'wave'], 
                action_types = ['kick', 'punch', 'squat', 'stand', 'wave'],
                
            )

    # -- Loop through all images
    ith_img = 1
    while ith_img <= images_loader.num_images:
        img, action_type = images_loader.load_next_image()
        image_disp = img.copy()

        print("\n\n========================================")
        print("\nProcessing {}/{}th image\n".format(ith_img, images_loader.num_images))

        # Detect skeleton
        humans = my_detector.detect(img)
        skelsInfo = SkeletonDetector.humans_to_skelsInfo(humans, action_type)
        for ith_skel in range(0, len(skelsInfo)):
            skeleton = SkeletonDetector.get_ith_skeleton(skelsInfo, ith_skel)

            # Classify action
            if DO_INFERENCE:
                prediced_label = classifier.predict(skeleton)
                print("prediced label is :", prediced_label)
            else:
                prediced_label = action_type
                print("Ground_truth label is :", prediced_label)

            if 1:
                # Draw skeleton
                if ith_skel == 0:
                    my_detector.draw(image_disp, humans)
                
                # Draw bounding box and action type
                myfunc.drawActionResult(image_disp, skeleton, prediced_label)

        # Write result to txt/png
        if SAVE_RESULTANT_SKELETON_TO_TXT_AND_IMAGE:
            myio.save_skeletons(SAVE_DETECTED_SKELETON_TO 
                + myfunc.int2str(ith_img, 5)+".txt", skelsInfo)
            cv2.imwrite(SAVE_DETECTED_SKELETON_IMAGES_TO 
                + myfunc.int2str(ith_img, 5)+".png", image_disp)

        if 1: # Display
            cv2.imshow("action_recognition", 
                cv2.resize(image_disp,(0,0),fx=1.5,fy=1.5))
            q = cv2.waitKey(1)
            if q!=-1 and chr(q) == 'q':
                break

        # Loop
        print("\n")
        ith_img += 1

