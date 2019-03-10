
'''
Input:
    skeletons/00001.txt ~ skeletons/xxxxx.txt
Output:
    skeletons_info.txt
'''

import numpy as np
from mylib.io import load_skeletons
import simplejson
import sys, os
import csv
CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"

if __name__=="__main__":
    # Parameters -------------------------------------------

    data_idx = "2"
    if data_idx == "1":
        NUM_SKELETONS = 2450
    else:
        NUM_SKELETONS = 3916

    read_from = "../skeleton_data/skeletons"+data_idx+"/"
    output_to = "../skeleton_data/skeletons"+data_idx+"_info.txt"
    output_to2 = "../skeleton_data/skeletons"+data_idx+"_info.csv"
    output_to2_good_only = "../skeleton_data/skeletons"+data_idx+"_info_good_only.csv"
    INVALID_VAL = 0 # Set not a number to this value
    DATA_AUGUMENT = False

    # Main -------------------------------------------

    int2str = lambda num, blank: ("{:0"+str(blank)+"d}").format(num)
    all_skeletons = []
    all_skeletons_good_only = []
    for i in range(1, NUM_SKELETONS+1):
        skeletons = load_skeletons(CURR_PATH + read_from + int2str(i, 5) + ".txt")
        idx_person = 0 # Only one person in each image
        skeleton = skeletons[idx_person]

        # change format to what I want
        sk = skeleton
        sk = sk[1:1+18*2] + [sk[0]] # 18 joints + action_type

        # -- Push to result list

        # 1. All data
        all_skeletons.append(sk.copy()) 

        # 2. Good data only
        LAST_JOINT_IDX = 14*2

        if INVALID_VAL not in sk[:LAST_JOINT_IDX]:
            all_skeletons_good_only.append(sk.copy())

        # 1. All data, + augumented data
        if DATA_AUGUMENT:
            if sk[-1] == "stand" and np.random.random() < 0.3:  # randomly drop feet when standing
                sk_aug = sk.copy()
                RIGHT_FOOT = 20
                LEFT_FOOT = 26
                sk_aug[RIGHT_FOOT] = 0
                sk_aug[RIGHT_FOOT+1] = 0
                sk_aug[LEFT_FOOT] = 0
                sk_aug[LEFT_FOOT+1] = 0
                all_skeletons.append(sk_aug.copy())
            if 1 and np.random.random() < 0.3: # randomly drop data
                idx_joint_to_drop = int(np.random.random()*LAST_JOINT_IDX/2)
                sk_aug = sk.copy()
                sk_aug[idx_joint_to_drop] = 0
                sk_aug[idx_joint_to_drop+1] = 0
                all_skeletons.append(sk_aug.copy())

    print("There are {} skeleton data.".format(len(all_skeletons)))
    print("There are {} good skeleton data.".format(len(all_skeletons_good_only)))

    with open(CURR_PATH + output_to, 'w') as f:
        simplejson.dump(all_skeletons, f)

    def wrote_to_csv(filepath, data):
        with open(filepath, 'w') as f:
            writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            first_row = ["nose_x","nose_y","neck_x","neck_y",
                "Rshoulder_x","Rshoulder_y","Relbow_x","Relbow_y","Rwrist_x","RWrist_y",
                "LShoulder_x","LShoulder_y","LElbow_x","LElbow_y","LWrist_x","LWrist_y",
                "RHip_x","RHip_y","RKnee_x","RKnee_y","RAnkle_x","RAnkle_y",
                "LHip_x","LHip_y","LKnee_x","LKnee_y","LAnkle_x","LAnkle_y",
                "REye_x","REye_y","LEye_x","LEye_y","REar_x","REar_y","LEar_x","LEar_y","class"]
            writer.writerow(first_row)
            for sk in data:
                writer.writerow(sk)
    wrote_to_csv(CURR_PATH + output_to2, all_skeletons)
    wrote_to_csv(CURR_PATH + output_to2_good_only, all_skeletons_good_only)