import glob
import shutil
import cv2
import os


files = glob.glob("*.png")
files.sort()


CURR_SPEED_720 = slice(473, 485, None), slice(1002, 1016, None)

for i in range(91):    
    ph = f"number_{i}.png"
    dst = ph#"test.png"
    img = cv2.imread(ph, 1)
    img = img[CURR_SPEED_720]
    cv2.imwrite(dst, img)
    
    #shutil.move(ph,dst)
