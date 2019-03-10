
import numpy as np
import cv2
import math

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

