import sys, os
import numpy as np
import subprocess

def readFeats(fpath):
    if not os.path.exists(fpath):
      dname = os.path.dirname(fpath)
      fname = os.path.basename(fpath)
      subprocess.call('cd ' + dname + '; tar xf ' + fname + '.tar.gz', shell=True)
    feats = np.loadtxt(fpath)
    subprocess.call('rm ' + fpath, shell=True)
    return feats

if __name__ == '__main__':
    main()

