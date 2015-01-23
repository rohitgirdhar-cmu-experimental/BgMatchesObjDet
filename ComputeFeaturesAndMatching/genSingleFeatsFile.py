#!/usr/bin/python

from readFeatures import readFeats
import os
import numpy as np 

def L2normalize(feats):
    l2norm = np.sum(np.abs(feats)**2, axis=-1)**(1./2)
    feats = feats / l2norm[:, None]
    return feats

def savetxt_compact(fname, x, fmt="%.9g", delimiter=' '):
    with open(fname, 'a+') as fh:
        for row in x:
            line = delimiter.join("0" if value == 0 else fmt % value for value in row)
            fh.write(line + '\n')

featsdir = '../tempdata/selsearch_feats/'
outfpath = '../tempdata/selsearch_feats_all_normalized.txt'
for i in range(1, 237 + 1):
    feats = np.array(readFeats(os.path.join(featsdir, str(i) + '.dat')))
    feats = L2normalize(feats)
    savetxt_compact(outfpath, feats)
    print 'Done for', i

