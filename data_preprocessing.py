import numpy as np


def head_reference(X):
    for i in range(len(X)):
        for j in range(1, int(len(X[i])/2)):
            X[i, j*2] = X[i, j*2] - X[i, 2]
            X[i, j*2+1] = X[i, j*2+1] - X[i, 3]
    return X

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
