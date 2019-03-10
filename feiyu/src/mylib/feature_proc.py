import numpy as np

# Math-----------------------------------------------
PI = np.pi
Inf = float("inf")

def pi2pi(x):
    if x>PI:
        x-=2*PI
    if x<=-PI:
        x+=2*PI
    return x

def calc_relative_angle(x1, y1, x0, y0, base_angle):
    # compute rotation from {base_angle} to {(x0,y0)->(x1,y1)}
    if (y1==y0) and (x1==x0):
        return 0
    a1 = np.arctan2(y1-y0, x1-x0)
    return pi2pi(a1 - base_angle)

def calc_relative_angle_v2(p1, p0, base_angle):
    return calc_relative_angle(p1[0], p1[1], p0[0], p0[1], base_angle)


# Feature selection/extraction/reduction -----------------------------------------------
class ProcFtr(object):
    
    @staticmethod
    def retrain_only_body_joints(x_input):
        x0 = x_input.copy()
        x0 = x0[2:2+13*2]
        return x0
    
    @staticmethod
    def normalize(x_input):
        lx = []
        ly = []
        N = len(x_input)
        i = 0
        while i<N:
            lx.append(x_input[i])
            ly.append(x_input[i+1])
            i+=2
        lx = np.array(lx)
        ly = np.array(ly)
        lx -= np.mean(lx)
        ly -= np.mean(ly)

        dlx = np.max(lx) - np.min(lx)
        lx /= 1 if dlx==0 else dlx

        dly = np.max(ly) - np.min(ly)
        ly /= 1 if dly==0 else dly
        
        # save it back to x
        x_new = []
        for i in range(int(N/2)):
            x_new.append(lx[i])
            x_new.append(ly[i])
        return x_new
    
    @staticmethod
    def joint_pos_2_angle(x_input):
        
        # First, manually select the following 13 joints:
        #    neck, l & r: [should, elbow, wrist], l & r: [hip, knee, ankle]
        x0 = x_input.copy()
        x0 = x0[2:2+13*2]
        
        N = len(x0)
        i = 0
        x = []
        while i<N:
            x.append(x0[i]) # x
            x.append(-x0[i+1]) # -y
            i+=2
            
        # ---------------------- Get joint positions ----------------------
        class Tmp(object):
            def __init__(self, x):
                self.x = x
                self.i = 0
            def get_next_point(self):
                p = [self.x[self.i], self.x[self.i+1]]
                self.i += 2
                return p
        tmp = Tmp(x)
        
        pneck = tmp.get_next_point()
        
        prshoulder = tmp.get_next_point()
        prelbow = tmp.get_next_point()
        prwrist = tmp.get_next_point()
        
        plshoulder = tmp.get_next_point()
        plelbow = tmp.get_next_point()
        plwrist = tmp.get_next_point()

        prhip = tmp.get_next_point()
        prknee = tmp.get_next_point()
        prankle = tmp.get_next_point()
        
        plhip = tmp.get_next_point()
        plknee = tmp.get_next_point()
        plankle = tmp.get_next_point()
        
        # ---------------------- Get joint angels ----------------------
                
        class Tmp2(object):
            def __init__(self):
                self.j = 0
                self.x_new = [Inf]*12 # 12 angles
            
            def set_next_angle(self, next_joint, base_joint, base_angle):
                angle=calc_relative_angle_v2(next_joint, base_joint, base_angle)
                self.x_new[self.j]=angle
                self.j+=1

        tmp2 = Tmp2()
        
        tmp2.set_next_angle(prshoulder, pneck, PI) # r-shoulder
        tmp2.set_next_angle(prelbow, prshoulder, -PI/2) # r-elbow
        tmp2.set_next_angle(prwrist, prelbow, -PI/2) # r-wrist
        
        tmp2.set_next_angle(plshoulder, pneck, 0) # l-shoulder
        tmp2.set_next_angle(plelbow, plshoulder, -PI/2) # l-elbow
        tmp2.set_next_angle(plwrist, plelbow, -PI/2) # l-wrist
        
        tmp2.set_next_angle(prhip, pneck, -PI/2-PI/18)
        tmp2.set_next_angle(prknee, prhip, -PI/2) 
        tmp2.set_next_angle(prankle, prknee, -PI/2)
        
        tmp2.set_next_angle(plhip, pneck, -PI/2+PI/18)
        tmp2.set_next_angle(plknee, plhip, -PI/2) 
        tmp2.set_next_angle(plankle, plknee, -PI/2)
        
        x_new = tmp2.x_new
        
        if 0:
            x_new = [val/PI*180 for val in x_new]
        
        # Return new features
        #print("x0:", x0)
        #print("x_res:", x_new)
        return x_new
    
    @staticmethod
    def pos2angles(x):
        None
        return

    @staticmethod
    def pose_normalization(x):
        NUM_JOINTS_XY = 13*2
        def retrain_only_body_joints(x_input):
            x0 = x_input.copy()
            x0 = x0[2:2+NUM_JOINTS_XY]
            return x0

        def normalize(x_input):
            # Separate original data into x_list and y_list
            lx = []
            ly = []
            N = len(x_input)
            i = 0
            while i<N:
                lx.append(x_input[i])
                ly.append(x_input[i+1])
                i+=2
            lx = np.array(lx)
            ly = np.array(ly)

            # Get rid of undetected data (=0)
            non_zero_x = []
            non_zero_y = []
            for i in range(int(N/2)):
                if lx[i] != 0:
                    non_zero_x.append(lx[i])
                if ly[i] != 0:
                    non_zero_y.append(ly[i])

            if len(non_zero_x)==0 or len(non_zero_y)==0:
                return np.array([0]*NUM_JOINTS_XY)

            # Normalization x/y data according to the bounding box
            origin_x = np.min(non_zero_x)
            origin_y = np.min(non_zero_y)
            len_x = np.max(non_zero_x) - np.min(non_zero_x)
            len_y = np.max(non_zero_y) - np.min(non_zero_y)
            
            len_x = 1 if len_x == 0 else len_x
            len_y = 1 if len_y == 0 else len_y

            x_new = []
            for i in range(int(N/2)):
                if (lx[i] + ly[i]) == 0:
                    x_new.append(-1)
                    x_new.append(-1)
                else:
                    x_new.append((lx[i] - origin_x) / len_x)
                    x_new.append((ly[i] - origin_y) / len_y)
            return np.array(x_new)

        x_body_joints_xy = retrain_only_body_joints(x)
        x_body_joints_xy = normalize(x_body_joints_xy)
        return x_body_joints_xy
