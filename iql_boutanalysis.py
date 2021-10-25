import h5py
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as pl
from scipy.signal import argrelmax, argrelmin
import sys

def load_bout_data():
    h5f = h5py.File("total_2rep_smooth_modelData.h5", 'r')
    return h5f

def grab_first(x):
    try:
        return next(x)
    except StopIteration:
        return -1

# everything inverted here so positive is right, neg is left.
    
def delta_yaw(heading):
    h0, h1 = heading
    if h1 - h0 > np.pi:
        dyaw = h0 + (2*np.pi - h1)
    elif h1 - h0 < -np.pi:
        dyaw = -1 * (h1 + (2*np.pi - h0))
    else:
        dyaw = -1*(h1 - h0)
    return dyaw

def bout_az(eyepos, heading):
    h0, h1 = heading
    eye1, eye2 = eyepos
    # unit vector pointing in direction of position change
    delta_eyepos_unit = (eye2 - eye1) / np.linalg.norm(eye2-eye1)
    # unit vector pointing in direction of heading
    heading_unit = [np.cos(h0), np.sin(h0)]
    bout_az = np.sign(np.cross(delta_eyepos_unit, heading_unit)) * np.arccos(
        np.dot(delta_eyepos_unit, heading_unit))
    return bout_az

def bout_dist(eyepos):
    eye1, eye2 = eyepos
    return np.linalg.norm(eye2-eye1)
    

class BoutDataCollector:
    def __init__(self, bout_indices):
        self.bout_length = 69
        self.num_segments = 15
        self.decimate_tail_by = 3
        self.decimate_time_by = 2
        self.segments_to_keep = range(0,
                                      self.num_segments,
                                      self.decimate_tail_by)
        self.bd = load_bout_data()
        self.bouts_of_interest = bout_indices
        self.bout_initiations = []
        self.tailangles = []
        self.find_bouts_and_tailangles()
        self.eye_positions = np.array(self.bd['position1'])[self.bouts_of_interest]
        self.heading_deltas = np.array(self.bd['heading'])[self.bouts_of_interest]
        

    def find_bouts_and_tailangles(self):
        tail_angles_raw = list(map(
            lambda t: np.arctan2(t[..., 1], t[..., 0]).T, self.bd["tailComponents"][
                self.bouts_of_interest]))
        bout_detector = lambda x: grab_first(filter(lambda y: np.abs(x[0])[y] > .3,
                                                    argrelmax(np.abs(x[0]))[0])) - 5
        self.bout_initiations = list(map(bout_detector, tail_angles_raw))
        self.tailangles = np.array([
            np.pad(
                b[:, bi:sys.maxsize:decimate_time_by],
                (0, np.ceil(
                    self.bout_length / self.decimate_time_by).astype(int) -
                 len(b[:, bi:sys.maxsize:self.decimate_time_by][0])),
                constant_values=np.nan)[
                    self.segments_to_keep] for b, bi in zip(
                    tail_angles_raw, self.bout_initiations)])

        
        
    def export_bout_dataframe(self):
        headers_tail = ['tailseg' + str(i) + 't' + str(j)
                        for i in self.segments_to_keep for j in range(
                                0, self.bout_length, self.decimate_time_by)]
        headers_bout = ['BoutAz', 'BoutYaw', 'BoutDistance']
        all_headers = headers_tail + headers_bout
        df = pd.DataFrame(columns=all_headers)
        for tailangles, eye_position, delta_heading in zip(self.tailangles,
                                                           self.eye_positions,
                                                           self.heading_deltas):
            b_az = bout_az(eye_position, delta_heading)
            b_yaw = delta_yaw(delta_heading)
            b_dist = bout_dist(eye_position)
            ta_concat = np.concatenate(tailangles).tolist()
            dict_entry = {h: val for h, val in zip(
                all_headers, ta_concat+[b_az, b_yaw, b_dist])}
            df.loc[len(df.index)] = dict_entry

        for c in df.columns:
            if len(df[c].value_counts()) == 0:
                del df[c]
        df.to_csv('boutdata.csv', index=False)  #, na_rep='NaN')
        return df






    
# if __name__ == "__main__":
#     random_indices = np.cumsum([np.random.randint(200) for i in range(1000)])
#     bd = BoutDataCollector(random_indices)
#     bd.export_bout_dataframe()



    
    
'''

tailComponents is arranged by bout, frame, 15 tail segments with an XY angle.
only relevant data pieces are tailComponents and deltaPosition

===========================================================================

            'tailComponents':
                                    <bout, frame, tail segment, angle components>
                                    Tail angles decomposed into xy components for each frame within 
                                    a bout, angles are with respect to heading (radians).
                                    =>	"tail segment" dimension: 
                                                    index [0] = tail tip, index [-1] = swim bladder 

            'deltaPosition':
                                    <bout, (Δx swimbladder, Δy swimbladder, Δx heading, Δy heading)>
                                    the change in global swimbladder position and the change in
                                    heading xy components between the start and end of each bout

            'position0':	<bout, start[0]/end[1] bout swimbladder positions, coordinates>
                                    global coordinates of swimbladder at the start and end
                                    of each bout

            'position1':	<bout, start[0]/end[0] bout third-eye positions, coordinates>
                                    global coordinates of "third-eye" at the start and end
                                    of each bout, third-eye = computed center of eyes

            'heading':		<bout, start[0]/end[1] bout heading> 
                                    heading in radians at the start and end of each bout

            'tailCoordintes': <bout, frame, tail segment, coordinates>
                                    local coordinates of tail segment during swim bouts
                                    =>	"tail segment" dimension: 
                                                    index [0] = tail tip, index [-1] = swim bladder 
'''
