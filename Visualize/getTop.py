import numpy as np
import os

inputdir = '../tempdata/dists_hash'
outputdir = '../tempdata/tops_hash/'
boxesdir = '../tempdata/selsearch_boxes/'
## Expects _posn files to be 0 indexed

def main():
    for i in range(1, 237 + 1):
        fpath = os.path.join(inputdir, str(i) + '.txt')
        f = open(fpath)
        scores = f.read().splitlines()
        f.close()
        scores = [float(s) for s in scores]
        scores = np.array(scores)
        order = np.argsort(scores)
        scores = scores[order]
        bboxes = readBboxes(os.path.join(inputdir, str(i) + "_posn.txt"), boxesdir)
        bboxes = bboxes[order]

        fout = open(outputdir + str(i) + ".txt", 'w')
        for j in range(100):
            if scores[j] >= 1.0: # no point after that
                continue
            fout.write('%d %f %f %f %f\n' % (order[j] + 1, bboxes[j][0], bboxes[j][1], bboxes[j][2], bboxes[j][3]))


def readBboxes(fpath_posn, dpath_act):
    f = open(fpath_posn)
    posns = [int(i) for i in f.read().splitlines()]
    res = []
    pos = 0
    for posn in posns:
        pos += 1
        line = getLine(dpath_act + str(pos) + ".txt", posn)
        res.append([float(i) for i in line.split(',')])
    f.close()
    return np.array(res)

def getLine(fpath, lno):
    with open(fpath) as f:
        for i, line in enumerate(f):
            if i == lno:
                return line.strip()

if __name__ == '__main__':
    main()

