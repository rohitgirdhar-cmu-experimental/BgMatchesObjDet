import sys, os
import numpy as np
import subprocess
from scipy.sparse import csr_matrix
sys.path.append('hashing/SparseLSH/')
from sparselsh import LSH

TEMPDATA = '/home/rgirdhar/work/03_temp/BgMatchesObjDet/tempdata/'
HASHDIR = os.path.join(TEMPDATA, 'selsearch_feats_hash')
OUTDIR = os.path.join(TEMPDATA, 'dists_hash')

def main():
    lsh = LSH(16, 9216, storage_config={'leveldb':{'db': os.path.join(HASHDIR, 'ldb')}},
            matrices_filename=os.path.join(HASHDIR, 'rand_planes.npz'))
    for i in range(1, 237 + 1):
        feats = readFeats(os.path.join(TEMPDATA, 'marked_feats', str(i) + '.dat'))
        assert np.shape(feats)[0] == 1 # for now at least
        matches = lsh.query(feats.getrow(0), distance_func='cosine')
        dists = np.ones(237)
        bbox_id = np.ones(237)
        for el in matches:
            dists[el[0] - 1] = 0
            bbox_id[el[0] - 1] = el[1]
#            feats2 = readFeats(os.path.join(TEMPDATA, 'selsearch_feats', str(el[0]) + '.dat'))
#            f2 = feats2.getrow(el[1])
#            import pdb
#            pdb.set_trace()
#            print f2.dot(feats.getrow(0))
        np.savetxt(os.path.join(OUTDIR, str(i) + '.txt'), dists, '%f', delimiter='\n')
        np.savetxt(os.path.join(OUTDIR, str(i) + '_posn.txt'), bbox_id, '%d', delimiter='\n')

def readFeats(fpath):
    dname = os.path.dirname(fpath)
    fname = os.path.basename(fpath)
    subprocess.call('cd ' + dname + '; tar xf ' + fname + '.tar.gz', shell=True)
    feats = csr_matrix(np.loadtxt(fpath))
    subprocess.call('rm ' + fpath, shell=True)
    return feats

if __name__ == '__main__':
    main()

