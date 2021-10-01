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

def delta_yaw(heading):
    h0, h1 = heading
    if h1 - h0 < -np.pi:
        dyaw = h0 + (2*np.pi - h1)
    elif h1 - h0 > np.pi:
        dyaw = -1 * (h1 + (2*np.pi - h0))
    else:
        dyaw = h1 - h0
    return dyaw

def bout_az(eyepos, heading):
    h0, h1 = heading
    eye1, eye2 = eyepos
    # unit vector pointing in direction of position change
    delta_eyepos_unit = (eye2 - eye1) / np.linalg.norm(eye2-eye1)
    # unit vector pointing in direction of heading
    heading_unit = [np.cos(h0), np.sin(h0)]
    bout_az = np.sign(np.cross(delta_eyepos_unit, heading_unit)) * np.arccos(np.dot(delta_eyepos_unit, heading_unit))
    return bout_az

def bout_dist(eyepos):
    eye1, eye2 = eyepos
    return np.linalg.norm(eye2-eye1)
    

# write some clever slicing methods to decimate this csv. its way too big. 

def make_dataframe(ta_at_b0, eyeposlist, fish_heading_angles, headers, clip_data):
    df = pd.DataFrame(columns=headers)
    for tailangles, eye_position, delta_heading in zip(ta_at_b0[0:clip_data],
                                                       eyeposlist[0:clip_data],
                                                       fish_heading_angles[0:clip_data]):
        b_az = bout_az(eye_position, delta_heading)
        b_yaw = delta_yaw(delta_heading)
        b_dist = bout_dist(eye_position)
        ta_concat = np.concatenate(tailangles).tolist()
        dict_entry = {h: val for h, val in zip(
            all_headers, ta_concat+[b_az, b_yaw, b_dist])}
        df.loc[len(df.index)] = dict_entry
    df.to_csv('boutdata.csv')
    return df

            

    
if __name__ == "__main__":
    random_indices = np.cumsum([np.random.randint(200) for i in range(1000)])
    bout_length = 69
    num_segments = 15
    decimate_tail_by = 3
    decimate_time_by = 2
    segments_to_keep = range(0, num_segments, decimate_tail_by)
    bd = load_bout_data()
    eye_positions = np.array(bd['position1'])[random_indices]
    heading_deltas = np.array(bd['heading'])[random_indices]
    atan_map = lambda t: np.arctan2(t[..., 1], t[..., 0]).T
  #  tail_angles = list(map(atan_map, bd["tailComponents"]))
    bout_detector = lambda x: grab_first(filter(lambda y: np.abs(x[0])[y] > .3,
                                                argrelmax(np.abs(x[0]))[0])) - 5
    bout_initiation = list(map(bout_detector, tail_angles))
    tail_angles_starting_at_b0 = np.array([
            np.pad(b[:, bi:sys.maxsize:decimate_time_by], (0, np.ceil(bout_length/decimate_time_by).astype(int) - len(b[:, bi:sys.maxsize:decimate_time_by][0])),
                   constant_values=np.nan)[segments_to_keep] for b, bi in zip(
                       tail_angles, bout_initiation)])[random_indices]
    # tailseg1_t1 -> t15,
    # have to decimate time by decreasing j to range(0, bout_length, decimate_time_by)
    headers_tail = ['tailseg' + str(i) + 't' + str(j) for i in segments_to_keep for j in range(0, bout_length, decimate_time_by)]
    
    headers_bout = ['BoutAz', 'BoutYaw', 'BoutDistance']
    all_headers = headers_tail + headers_bout

    df = make_dataframe(tail_angles_starting_at_b0,
                        eye_positions, heading_deltas, all_headers, sys.maxsize)
        
        
        
    # part of me wants to map the rotations and displacements to my original coordinate system.
    # bout az, bout yaw, and bout dist. 

    
   
    
    
    
    
    




# tailComponents is arranged by bout, frame, 15 tail segments with an XY angle.
# only relevant data pieces are tailComponents and deltaPosition

'''	
==> Load '*_modelData.h5':
	`from andrewHarvard import loadNNData`
	`tailComponents, deltaPosition, position0, position1, heading, tailCoordinates = loadNNData(file_path)`
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
