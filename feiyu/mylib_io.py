import simplejson
import cv2

FILENAME_IMAGES_INFO = "images_info.txt"

def save_images_info(path, images_info):
    with open(path+FILENAME_IMAGES_INFO, 'w') as f:
        simplejson.dump(images_info, f)

def load_images_info(path):
    print(path+FILENAME_IMAGES_INFO)
    with open(path+FILENAME_IMAGES_INFO, 'r') as f:
        images_info = simplejson.load(f)
        return images_info
    return None

def save_skeletons(filename, skeletons):
    # skeleton = [action_type (optional), 18*[x,y], 18*score], length = 1+36+18=55
    with open(filename, 'w') as f:
        simplejson.dump(skeletons, f)

def load_skeletons(filename):
    # skeleton = [action_type (optional), 18*[x,y], 18*score], length = 1+36+18=55
    with open(filename, 'r') as f:
        skeletons = simplejson.load(f)
        return skeletons
    return None

def print_images_info(images_info):
    for img_info in images_info:
        print(img_info)

def int2str(num, blank):
    return ("{:0"+str(blank)+"d}").format(num)

def int2name(num):
    return int2str(num, 5)+".png"

def collect_images_info_from_source_images(path, valid_images_txt):
    images_info = list()

    with open(path + valid_images_txt) as f:

        folder_name = None
        action_type = None
        cnt_action = 0
        actions = set()
        cnt_clip = 0
        cnt_image = 0

        for cnt_line, line in enumerate(f):

            if line.find('_') != -1:  # A new video type
                folder_name = line[:-1]
                action_type = folder_name.split('_')[0]
                if action_type not in actions:
                    cnt_action += 1
                    actions.add(action_type)

            elif len(line) > 1:  # line != "\n"
                # print("Line {}, len ={}, {}".format(cnt_line, len(line), line))
                indices = [int(s) for s in line.split()]
                idx_start = indices[0]
                idx_end = indices[1]
                cnt_clip += 1
                for i in range(idx_start, idx_end+1):
                    filepath = folder_name+"/"+int2name(i)
                    cnt_image += 1
                    
                    # Save: 5 values
                    d = [cnt_action,cnt_clip, cnt_image, action_type, filepath]
                    images_info.append(d)
                    # An example: [8, 49, 2687, 'wave', 'wave_03-02-12-35-10-194/00439.png']
    return images_info
    '''
    Other notes
    { read line:
        line = fp.readline()
        if fail, then line == None
    }
    '''

class ImageLoader(object):
    # [1, 7, 54, "jump", "jump_03-02-12-34-01-795/00240.png"]
    def __init__(self, images_info, TRAINING_IMAGES_PATH):
        self.imgs_path = TRAINING_IMAGES_PATH
        self.images_info = images_info
        self.num_images = len(self.images_info)

    def imread(self, index):
        return cv2.imread(self.imgs_path + self.get_filename(index))
    
    def get_filename(self, index):
        return self.images_info[index-1][4]

    def get_action_type(self, index):
        return self.images_info[index-1][3]

        
if __name__=="__main__":
    '''
    For test case, see "images_info_save_to_file.py"
    '''
