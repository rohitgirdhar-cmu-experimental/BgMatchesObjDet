import sys, os
import numpy as np
import subprocess
import time
from scipy.sparse import csr_matrix
sys.path.append('hashing/SparseLSH/')
from sparselsh import LSH

TEMPDATA = '/home/rgirdhar/work/03_temp/BgMatchesObjDet/tempdata/'
HASHDIR = os.path.join(TEMPDATA, 'selsearch_feats_hash')
try:
    s.makedirs(HASHDIR)
except:
    pass

def main():
    lsh = LSH(10, 9216, storage_config={'leveldb':{'db': os.path.join(HASHDIR, 'ldb')}},
            matrices_filename=os.path.join(HASHDIR, 'rand_planes.npz'))
    for i in range(1, 237 + 1):
        stime = time.time()
        feats = readFeats(os.path.join(TEMPDATA, 'selsearch_feats', str(i) + '.dat'))
        rtime = time.time()
        for ifeat in range(feats.shape[0]):
            feat = feats.getrow(ifeat)
            lsh.index(feat, extra_data=(i, ifeat))
        print('Indexed image %d (%d feats) in %f (read) + %f (index) = %f seconds' %
                (i,
                 feats.shape[0],
                 rtime - stime,
                 time.time() - rtime,
                 time.time() - stime))

def readFeats(fpath):
    dname = os.path.dirname(fpath)
    fname = os.path.basename(fpath)
    subprocess.call('cd ' + dname + '; tar xf ' + fname + '.tar.gz', shell=True)
    feats = csr_matrix(np.loadtxt(fpath))
    subprocess.call('rm ' + fpath, shell=True)
    return feats

if __name__ == '__main__':
    main()

