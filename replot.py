# Created by Baole Fang at 4/23/23

import os
import numpy as np
from learning import plot
import matplotlib.pyplot as plt

def replot(path,start,end):
    lst=[os.path.join(path,name) for name in os.listdir(path)]
    lst.sort(key=os.path.getmtime)
    for filename in lst:
        if filename.endswith('.npy'):
            acc=np.load(filename)
            acc=acc[:,start:end]
            label, base, samples, batch=os.path.basename(filename).rstrip('.npy').split('-')
            plot(acc, label, int(base)+start*int(batch), int(batch))


if __name__ == '__main__':
    root='outputs/SILVA_138_3_8_sliding_0/xgboost'
    start=0
    end=-1
    replot(root,start,end)
    plt.title('xgboost')
    plt.tight_layout()
    plt.savefig(os.path.join(root,'result_default_{}_{}.png'.format(start,end)))
    plt.close()
