
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

import mylib_io as myio

from keras.models import load_model

# PATHS ==============================================================

CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
print(CURR_PATH)
DATA_FOLDER = "/home/feiyu/Desktop/C1/FinalProject/data/source_images/"

# Openpose ==============================================================

sys.path.append("/home/feiyu/Desktop/C1/FinalProject/src/githubs/tf-pose-estimation")
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



# my functions ==============================================================


def drawLineToImage(img_display, p0, p1, color=None, line_width=2):
    if color is None:
        color=[0,0,255] # red
    if type(p0)==list:
        p0=(p0[0],p0[1])
    if type(p1)==list:
        p1=(p1[0],p1[1])
    cv2.line(img_display,p0,p1,color,line_width)

def drawBoxToImage(img_display, p0, p1, color=None, line_width=2):
    x1=p0[0]
    y1=p0[1]
    x2=p1[0]
    y2=p1[1]
    xs=[x1,x2,x2,x1,x1]
    ys=[y1,y1,y2,y2,y1]

    colors=['r','g','b']
    colors_dict={'b':[255,0,0],'g':[0,255,0],'r':[0,0,255]}
    if color==None:
        color=colors_dict['r']
    if type(color)!=list:
        color=colors_dict[color]

    for i in range(4):
        drawLineToImage(img_display, (xs[i],ys[i]),  (xs[i+1],ys[i+1]), color, line_width=2)

def drawActionResult(img_display, skeleton, action_result):
    font = cv2.FONT_HERSHEY_SIMPLEX 

    minx = 999
    miny = 999
    maxx = -999
    maxy = -999
    for i in range(18):
        minx = np.min(minx, skeleton[i])
        maxx = np.min(maxx, skeleton[i])
        miny = np.min(miny, skeleton[i+1])
        maxy = np.min(maxy, skeleton[i+1])
    
    minx = minx * img_display.shape[1]
    miny = miny * img_display.shape[0]
    maxx = maxx * img_display.shape[1]
    maxy = maxy * img_display.shape[0]
    
    # Draw bounding box
    drawBoxToImage(img_display, [minx, miny], [maxx, maxy])

    # Draw text at left corner
    # TEST_ROW = miny 
    # TEST_COL = minx
    TEST_ROW = 300 
    TEST_COL = 300
    img_display = cv2.putText(
        img_display, action_result, (TEST_COL, TEST_ROW), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

# FUNCTIONS ==============================================================

int2str = lambda num, blank: ("{:0"+str(blank)+"d}").format(num)

class DetectSkeleton(object):
    # This func is copied from https://github.com/ildoonet/tf-pose-estimation

    def __init__(self, model="mobilenet_thin"):
        models = set({"mobilenet_thin", "cmu"})
        self.model = model if model in models else "mobilenet_thin"

        parser = argparse.ArgumentParser(description='tf-pose-estimation run')
        # parser.add_argument('--image', type=str, default='./images/p1.jpg')
        # parser.add_argument('--model', type=str, default='cmu', help='cmu / mobilenet_thin')

        # parser.add_argument('--resize', type=str, default='0x0',
        #                     help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
        parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                            help='if provided, resize heatmaps before they are post-processed. default=1.0')

        args = parser.parse_args()

        # w, h = model_wh(args.resize)
        w, h = model_wh("432x368")
        if w == 0 or h == 0:
            e = TfPoseEstimator(get_graph_path(self.model),
                                target_size=(432, 368))
        else:
            e = TfPoseEstimator(get_graph_path(self.model), target_size=(w, h))

        self.args = args
        self.w, self.h = w, h
        self.e = e
        self.fps_time = time.time()

    def detect(self, image):
        t = time.time()

        # Inference
        humans = self.e.inference(image, resize_to_default=(self.w > 0 and self.h > 0),
                                  upsample_size=self.args.resize_out_ratio)

        # Print result and time cost
        print("humans:", humans)
        elapsed = time.time() - t
        logger.info('inference image in %.4f seconds.' % (elapsed))

        return humans


    def draw_heatmaps(self, image, humans):  # This draws a single image
        img_disp = image.copy()
        img_disp = TfPoseEstimator.draw_humans(img_disp, humans, imgcopy=False)

        import matplotlib.pyplot as plt

        fig = plt.figure()
        a = fig.add_subplot(2, 2, 1)
        a.set_title('Result')
        plt.imshow(cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB))

        bgimg = cv2.cvtColor(img_disp.astype(np.uint8), cv2.COLOR_BGR2RGB)
        bgimg = cv2.resize(
            bgimg, (self.e.heatMat.shape[1], self.e.heatMat.shape[0]), interpolation=cv2.INTER_AREA)

        # show network output
        a = fig.add_subplot(2, 2, 2)
        plt.imshow(bgimg, alpha=0.5)
        tmp = np.amax(self.e.heatMat[:, :, :-1], axis=2)
        plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.5)
        plt.colorbar()

        tmp2 = self.e.pafMat.transpose((2, 0, 1))
        tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)
        tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)

        a = fig.add_subplot(2, 2, 3)
        a.set_title('Vectormap-x')
        # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
        plt.imshow(tmp2_odd, cmap=plt.cm.gray, alpha=0.5)
        plt.colorbar()

        a = fig.add_subplot(2, 2, 4)
        a.set_title('Vectormap-y')
        # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
        plt.imshow(tmp2_even, cmap=plt.cm.gray, alpha=0.5)
        plt.colorbar()
        plt.show()
    
    def draw(self, image, humans):
        img_disp = image.copy()
        img_disp = TfPoseEstimator.draw_humans(img_disp, humans, imgcopy=False)

        logger.debug('show+')
        cv2.putText(img_disp,
                    "FPS: %f" % (1.0 / (time.time() - self.fps_time)),
                    (20, 20),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)
        self.fps_time = time.time()

        return img_disp

    @staticmethod
    def humans2skeletons(humans, action_type=None):
        # skeleton = [action_type (optional), 18*[x,y], 18*score]
        skeletons = []
        for human in humans:
            if action_type is None:
                idx0 = 0
            else:
                idx0 = 1
            skeleton = [-1]*(idx0+18*2+18)
            if action_type is not None:
                skeleton[0] = action_type
            # for i, body_part in human.body_parts.iteritems():
            for i, body_part in human.body_parts.items(): # iterate dict
                idx = body_part.part_idx
                skeleton[idx0+2*idx]=body_part.x
                skeleton[idx0+2*idx+1]=body_part.y
                skeleton[idx0+36+idx]=body_part.score

            skeletons.append(skeleton)
        return skeletons


# ==============================================================

if __name__ == "__main__":

    loader = myio.ImageLoader(myio.load_images_info(CURR_PATH), DATA_FOLDER)
    
    detector = DetectSkeleton()
    action_recognition_model = load_model(CURR_PATH + "Skeleton-Based-Human-Action-Recognition/" + "action_recognition.h5")
        

    window_name = "action_recognition"
    for i in range(150, loader.num_images+1):

        img = loader.imread(i)
        filename = loader.get_filename(i)
        action_type = loader.get_action_type(i)
        print("\n\n========================================")
        print("\nProcessing {}, {}: {}\n".format(i, action_type, filename))

        if 0:
            cv2.imshow(window_name, img)
            cv2.waitKey()

        # Detect
        humans = detector.detect(img)

        # Write skeleton data to file
        skeletons = DetectSkeleton.humans2skeletons(humans, action_type)
        # myio.save_skeletons("skeletons/"+int2str(i, 5)+".txt", skeletons)

        
        # Draw and write display image to file
        image_disp = detector.draw(img, humans)
        cv2.imwrite("skeletons_images/"+int2str(i, 5)+".png", image_disp)
      

        # 识别
        skeleton = np.array(skeletons[0][1:1+18*2]).reshape(-1, 36)
 
        predict_number = np.argmax(action_recognition_model.predict(skeleton))
        action_dict = {
            1: "stand", 2: "walk", 3:"wave", 0:"squat"
        }
        action_result = action_dict[predict_number]

        print("\n\n ================================\n\n")
        print("action_result is :", action_result, "\n\n")
        drawActionResult(image_disp, skeleton,   action_result)

        # Display
        cv2.imshow(window_name, image_disp)
        q = cv2.waitKey(1)
        if q!=-1 and chr(q) == 'q':
            break

