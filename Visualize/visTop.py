# Generate visualization for the top elements from getTop

import sys, os
sys.path.append("/home/rgirdhar/libs/opencv/lib/python2.7/dist-packages")
import cv2
import numpy as np

imgsdir = "../dataset/PeopleAtLandmarks/corpus/"
topsdir = '../tempdata/tops_hash'
outimgspath = "../tempdata/tops_hash_vis/"

def main():
    with open("../dataset/PeopleAtLandmarks/ImgsList.txt") as f:
        lst = f.read().splitlines()
    for i in range(1, 237 + 1):
        out_dpath = outimgspath + str(i) + "/"
        I = cv2.imread(imgsdir + lst[i - 1])
        qcls = getClass(lst[i - 1])
        with open("../tempdata/marked_boxes/" + str(i) + ".txt") as fid:
            box = fid.readline().strip().split(',')
        qbox = [int(float(el)) for el in box]
        cv2.rectangle(I, (qbox[1], qbox[0]), (qbox[3], qbox[2]), (0,255,0), 3)
        I = cv2.resize(I, (256, np.shape(I)[0] * 256 / np.shape(I)[1]))
        cv2.imwrite(out_dpath + "q.jpg", I)

        topimgs, bboxes = readTopList(os.path.join(topsdir, str(i) + ".txt"))
        j = 0
        hitornot = []
        if not os.path.exists(out_dpath):
            os.makedirs(out_dpath)
        for topimg in topimgs:
            J = cv2.imread(imgsdir + lst[topimg - 1])
            tcls = getClass(lst[topimg - 1])
            # bbox are in sel search format 
            cv2.rectangle(J, (int(bboxes[j][1]), int(bboxes[j][0])), 
                    (int(bboxes[j][3]), int(bboxes[j][2])), (0,0,255), 3)
            J = cv2.resize(J, (256, np.shape(J)[0] * 256 / np.shape(J)[1]))
            cv2.imwrite(out_dpath + str(j) + ".jpg", J)

            if qcls == tcls:
                hitornot.append(1)
            else:
                hitornot.append(0)
            j += 1
        
        f = open(out_dpath + 'match.txt', 'w')
        f.write('\n'.join([str(el) for el in hitornot]))
        f.close()


def readTopList(fpath):
    with open(fpath) as f:
        lines = f.read().splitlines()
        lines = [line.split() for line in lines]
        topimgs = [int(line[0]) for line in lines]
        bboxes = [line[1:] for line in lines]
        bboxes = [[float(i) for i in bbox] for bbox in bboxes]
    return (topimgs, bboxes)

def getClass(fpath):
    dpath = os.path.dirname(fpath)
    return os.path.basename(dpath)

if __name__ == '__main__':
    main()
