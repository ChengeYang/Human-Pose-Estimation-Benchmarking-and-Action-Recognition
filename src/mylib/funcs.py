
import numpy as np
import cv2
import math

# drawings ==============================================================

def drawActionResult(img_display, skeleton, str_action_type):
    font = cv2.FONT_HERSHEY_SIMPLEX 

    minx = 999
    miny = 999
    maxx = -999
    maxy = -999
    i = 0
    NaN = 0

    while i < len(skeleton):
        if not(skeleton[i]==NaN or skeleton[i+1]==NaN):
            minx = min(minx, skeleton[i])
            maxx = max(maxx, skeleton[i])
            miny = min(miny, skeleton[i+1])
            maxy = max(maxy, skeleton[i+1])
        i+=2

    minx = int(minx * img_display.shape[1])
    miny = int(miny * img_display.shape[0])
    maxx = int(maxx * img_display.shape[1])
    maxy = int(maxy * img_display.shape[0])
    print(minx, miny, maxx, maxy)
    
    # Draw bounding box
    # drawBoxToImage(img_display, [minx, miny], [maxx, maxy])
    img_display = cv2.rectangle(img_display,(minx, miny),(maxx, maxy),(0,255,0), 4)

    # Draw text at left corner


    box_scale = max(0.5, min(2.0, (1.0*(maxx - minx)/img_display.shape[1] / (0.3))**(0.5) ))
    fontsize = 1.5 * box_scale
    linewidth = int(math.ceil(3 * box_scale))

    TEST_COL = int( minx + 5 * box_scale)
    TEST_ROW = int( miny - 10 * box_scale)

    img_display = cv2.putText(
        img_display, str_action_type, (TEST_COL, TEST_ROW), font, fontsize, (0, 0, 255), linewidth, cv2.LINE_AA)



int2str = lambda num, blank: ("{:0"+str(blank)+"d}").format(num)

# Test of some data proprocessing methods ==============================================================

def remove_head_offset(skeleton0):
    skeleton = skeleton0.copy()
    for i in range(1, 18):
        skeleton[i*2] -= skeleton[0]
        skeleton[i*2+1] -= skeleton[1]
    return skeleton


def pos2angles(sk):

    def get(i):
        return sk[i*2:i*2+2]

    # Calculate the joint angles for elbow, hip and knee
    def calc_angle(p1,p2,p3):
        dp1 = p1-p2
        dp2 = p3-p2
        if np.linalg.norm(p1)*np.linalg.norm(p2)*np.linalg.norm(p3) == 0:
            return -1
        # elif np.linalg.norm(dp1)*np.linalg.norm(dp2) == 0:
        #     return 0
        else:
            res = dp1.dot(dp2)/(np.linalg.norm(dp1)*np.linalg.norm(dp2))
            return res

    # Claculate angles for shouder
    def calc_angle_shoulder(p1,p2,p3):
        dp1 = p1-p2
        dp2 = p3-p2
        if np.linalg.norm(p1)*np.linalg.norm(p2)*np.linalg.norm(p3) == 0:
            return 0
        # elif np.linalg.norm(dp1)*np.linalg.norm(dp2) == 0:
        #     return 0
        else:
            res = dp1.dot(dp2)/(np.linalg.norm(dp1)*np.linalg.norm(dp2))
            return res


    p_neck = get(1)

    p_r_shoulder =  get(2)
    p_r_elbow =  get(3)
    p_r_wrist =  get(4)
    a_r_shoulder = calc_angle_shoulder(p_neck, p_r_shoulder, p_r_elbow)
    a_r_elbow = calc_angle(p_r_shoulder, p_r_elbow, p_r_wrist)

    p_l_shoulder =  get(5)
    p_l_elbow =  get(6)
    p_l_wrist =  get(7)
    a_l_shoulder = calc_angle_shoulder(p_neck, p_l_shoulder, p_l_elbow)
    a_l_elbow = calc_angle(p_l_shoulder, p_l_elbow, p_l_wrist)

    p_r_hip = get(8)
    p_r_knee = get(9)
    p_r_ankle = get(10)
    a_r_hip = calc_angle(p_neck, p_r_hip, p_r_knee)
    a_r_knee = calc_angle(p_r_hip, p_r_knee, p_r_ankle)

    p_l_hip = get(11)
    p_l_knee = get(12)
    p_l_ankle = get(13)
    a_l_hip = calc_angle(p_neck, p_l_hip, p_l_knee)
    a_l_knee = calc_angle(p_l_hip, p_l_knee, p_l_ankle)

    angles = [a_r_shoulder, a_r_elbow, a_l_shoulder, a_l_elbow, a_r_hip, a_r_knee, a_l_hip, a_l_knee]
    return np.array(angles)