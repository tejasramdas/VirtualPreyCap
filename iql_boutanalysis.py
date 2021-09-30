import h5py
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as pl
from scipy.signal import argrelmax, argrelmin

def load_bout_data():
    h5f = h5py.File("total_2rep_smooth_modelData.h5", 'r')
    return h5f

def grab_first(x):
    try:
        return next(x)
    except StopIteration:
        return -1
        
    

    
if __name__ == "__main__":
    bout_length = 69
    num_segments = 15
    bd = load_bout_data()
    atan_map = lambda t: np.arctan2(t[..., 1], t[..., 0]).T
    bout_angles = list(map(atan_map, bd["tailComponents"]))
    bout_detector = lambda x: grab_first(filter(lambda y: np.abs(x[0])[y] > .3, argrelmax(np.abs(x[0]))[0])) - 5
    bout_initiation = list(map(bout_detector, bout_angles))
    bout_angles_starting_at_b0 = [
            np.pad(b[:, bi:], (0, 69-len(b[:, bi:][0])),
                   constant_values=np.nan)[0:num_segments] for b, bi in zip(bout_angles, bout_initiation)]

    
    # tailseg1_t1 -> t15, 
    headers = ['tailseg' + str(i) + 't' + str(j) for i in range(num_segments) for j in range(bout_length)]

    # part of me wants to map the rotations and displacements to my original coordinate system.
    # bout az, bout yaw, and bout dist. 

    
    df = pd.DataFrame()
    
    
    
    
    




# tailComponents is arranged by bout, frame, 15 tail segments with an XY angle.
# only relevant data pieces are tailComponents and deltaPosition
"""

'tailComponents':
     <bout, frame, tail segment, angle components>
     Tail angles decomposed into xy components for each frame within 
     a bout, angles are with respect to heading (radians).
     =>	"tail segment" dimension: 
     index [0] = tail tip, index [-1] = swim bladder 


by xy components, i think it means the fish is oriented on the x axis, and 
the 


'deltaPosition':
     <bout, (Δx swimbladder, Δy swimbladder, Δx heading, Δy heading)>
     the change in global swimbladder position and the change in
     heading xy components between the start and end of each bout

"""
