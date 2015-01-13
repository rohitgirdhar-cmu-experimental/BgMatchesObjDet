# Generate visualization for the top elements from getTop

import sys, os
sys.path.append("/home/rgirdhar/libs/opencv/lib/python2.7/dist-packages")
import cv2
import numpy as np

imgsdir = "../dataset/PeopleAtLandmarks/corpus/"

def main():
    with open("../dataset/PeopleAtLandmarks/ImgsList.txt") as f:
        lst = f.read().splitlines()
    for i in range(1, 50):
        I = cv2.imread(imgsdir + lst[i])

        topimgs, bboxes = readTopList("../tempdata/tops/" + str(i) + ".txt")
        j = 0
        out_dpath = "../tempdata/tops_vis/" + str(i) + "/"
        if not os.path.exists(out_dpath):
            os.makedirs(out_dpath)
        for topimg in topimgs:
            J = cv2.imread(imgsdir + lst[topimg])
            cv2.rectangle(J, (int(bboxes[j][1]), int(bboxes[j][0])), 
                    (int(bboxes[j][3]), int(bboxes[j][2])), (0,0,255))
            cv2.imwrite(out_dpath + str(j) + ".jpg", J)
            j += 1
        

def readTopList(fpath):
    with open(fpath) as f:
        lines = f.read().splitlines()
        lines = [line.split() for line in lines]
        topimgs = [int(line[0]) for line in lines]
        bboxes = [line[1:] for line in lines]
        bboxes = [[float(i) for i in bbox] for bbox in bboxes]
    return (topimgs, bboxes)

if __name__ == '__main__':
    main()
