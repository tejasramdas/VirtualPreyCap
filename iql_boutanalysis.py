import h5py
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as pl
from scipy.signal import argrelmax, argrelmin
import sys
import json
import sppl.compilers.spn_to_dict as spn_to_dict
from sppl.transforms import Identity as I
from collections import OrderedDict

# next step is to get the validation set into a dataframe (last 3600 bouts)
# loop through each and assign a probability


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
    def __init__(self, bout_indices, df_id):
        self.bout_length = 69
        self.num_segments = 15
        self.decimate_tail_by = 3
        self.decimate_time_by = 2
        self.df_id = df_id
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
                b[:, bi:sys.maxsize:self.decimate_time_by],
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
        df.to_csv(self.df_id + '_boutdata.csv', index=False)  #, na_rep='NaN')
        return df


class SpplSampler:
    def __init__(self):
#        self.df = pd.read_csv("/Users/nightcrawler/inferenceql.auto-modeling/data/nullified.csv")
        self.df = pd.read_csv("/home/andrewbolton/inferenceql.auto-modeling/data/nullified.csv")
        self.model = self.loader()
#        with open('/Users/nightcrawler/inferenceql.auto-modeling/data/dep-prob.json', 'r') as f:
#            self.dep_prob = json.load(f)
        with open('/home/andrewbolton/inferenceql.auto-modeling/data/dep-prob.json', 'r') as f:
            self.dep_prob = json.load(f)
        self.dep_df = pd.DataFrame()
        self.extract_dependencies()
        self.conditioned_model = self.model
#        self.observed_data = 

    def loader(self):
#        with open("/Users/nightcrawler/inferenceql.auto-modeling/data/sppl/merged.json") as f:
        with open("/home/andrewbolton/inferenceql.auto-modeling/data/sppl/merged.json") as f:        
            spn_dict = json.load(f)
        model = spn_to_dict.spn_from_dict(spn_dict)
        return model

    def plot_dependencies_in_df_order(self):
        c1s = []
        ps = []
        for c1 in self.dep_prob.keys():
            for c2 in self.dep_prob.keys():
                if c1 == c2:
                    p = 1.
                else:
                    p = self.dep_prob[c1][c2]
                ps.append(p)
            c1s.append(c1)
        ps_arr = np.array(ps)
        ps_reshape = ps_arr.reshape((int(np.sqrt(len(ps))), int(np.sqrt(len(ps)))))
        dep_df = pd.DataFrame(ps_reshape, index=c1s, columns=c1s)   #, index=c1s, columns=c2s)   
        print(dep_df)
        f = pl.figure(figsize=(6,6))
        sns.set(font_scale = .1)
        sns.heatmap(dep_df, xticklabels=1, yticklabels=1, cmap="viridis")
        pl.savefig("deps_in_df_order.pdf")

    def extract_dependencies(self):
        c1s = []
        c2s = []
        ps = []
        for c1 in self.dep_prob.keys():
            for c2 in self.dep_prob.keys():
                if c1 == c2:
                    p = 1.
                else:
                    p = self.dep_prob[c1][c2]
                c1s.append(c1)
                c2s.append(c2)
                ps.append(p)
        self.dep_df = pd.DataFrame({"c1": c1s, "c2": c2s, "p": ps})

    def plot_clustermap(self, ax=None, **kwargs):
        """Plot a clustermap by pivoting the last 3 columns of `df`.
        """
        if len(df.columns) < 3:
            raise ValueError('At least three columns requried: %s' % (self.dep_df.columns,))
        # Pivot the matrix.
        pivot = self.dep_df.pivot(
            index=self.dep_df.columns[-3],
            columns=self.dep_df.columns[-2],
            values=self.dep_df.columns[-1],
        )
        pivot.fillna(0, inplace=True)
        # Check if all values are between 0 and 1 to set vmin and vmax.
        (vmin, vmax) = (None, None)
        if all(0 <= v <= 1 for v in self.dep_df.iloc[:,-1]):
            (vmin, vmax) = (0, 1)
        D = np.asmatrix(pivot.values) 
        zmatrix = _clustermap(D,
                              xticklabels=pivot.columns.tolist(),
                              yticklabels=pivot.index.tolist(),
                              vmin=vmin,
                              vmax=vmax
                              )
        # Heuristics for the size.
        figsize = kwargs.pop('figsize', None)
        if figsize is None:
            half_root_col = (self.dep_df.shape[0] ** .5) / 2.
            figsize = (half_root_col, .8 * half_root_col)
        zmatrix.fig.set_size_inches(figsize)
        return zmatrix

    def plot_depedency_heatmap(self, ax=None, **kwargs):
        """Plot a heatmap by pivoting the last 3 columns of `df`.
        """
        if len(self.dep_df.columns) < 3:
            raise ValueError('At least three columns requried: %s' % (self.dep_df.columns,))
        # Pivot the matrix.
        pivot = self.dep_df.pivot(
            index=self.dep_df.columns[-3],
            columns=self.dep_df.columns[-2],
            values=self.dep_df.columns[-1],
        )
        pivot.fillna(0, inplace=True)
        # Check if all values are between 0 and 1 to set vmin and vmax.
        (vmin, vmax) = (None, None)
        if all(0 <= v <= 1 for v in self.dep_df.iloc[:,-1]):
            (vmin, vmax) = (0, 1)
        # Apply the optimal ordering from a clustermap.
        D = np.asmatrix(pivot.values)
        (xordering, yordering) = _clustermap_ordering(D)
        xticklabels = np.asarray(pivot.columns)[xordering]
        yticklabels = np.asarray(pivot.index)[yordering]
        D = D[:, xordering]
        D = D[yordering, :]
        ax = sns.heatmap(
            D,
            xticklabels=xticklabels,
            yticklabels=yticklabels,
            linewidths=0.2,
            cbar=kwargs.get('cbar', True),
            cmap='viridis',
            ax=ax,
            vmin=vmin,
            vmax=vmax,
        )
        # Heuristics for the size.
     
        figsize = kwargs.pop('figsize', None)
        if figsize is None:
            half_root_col = (self.dep_df.shape[0] ** .5) / 2.5
            figsize = (half_root_col, .8 * half_root_col)
        ax.get_figure().set_size_inches(figsize)
        return ax

    # this takes a couple minutes to sample! 
    def generate(self, N, *constraint_row):
        if constraint_row == ():
            model = self.model
        else:
            self.make_conditional_spn(constraint_row[0])
            model = self.conditioned_model
        samples = model.sample(N)
        return pd.DataFrame(
            [
                {k.__str__(): v
                 for k, v in sample.items()
                 if ('_cluster' not in k.__str__()) and k.__str__()!='child'
                 }
                for sample in samples
            ])[self.df.columns]

    # get out by df.loc[1]
    Def make_conditional_spn(self, df_row):
        constraints = {I(k): v for k, v in df_row.to_dict().items() if (
            k[0:4] != "Bout" and np.isfinite(v))}
        self.conditioned_model = self.model.constrain(constraints)


def posterior_3D_boutplot(df, groundtruth):
    fig = pl.figure()
    ax = fig.add_subplot(projection='3d')
    density = ax.scatter(df["BoutAz"], df["BoutYaw"], df["BoutDistance"],
                         c='k', alpha=.1)
    groundtruth = ax.scatter(groundtruth[0], groundtruth[1], groundtruth[2], color='r')
    ax.set_xlabel('Bout Az')
    ax.set_ylabel('Bout Yaw')
    ax.set_zlabel('Bout Distance')
    pl.show()


def test_model_v_groundtruth(sp_sampler, df_row, plot_dist):
    groundtruth = [np.array(df_row["BoutAz"]),
                   np.array(df_row["BoutYaw"]),
                   np.array(df_row["BoutDistance"])]
    # here will be a generate call that will take a min to make a SPN. for now
    # placehold with a sample call

    angle_bound = np.deg2rad(1)
    dist_bound = .1
#    posterior_dist = sp_sampler.generate(100, df_row)
    posterior_dist = sp_sampler.generate(100)
    print(groundtruth)
    prob_gt_under_model = sp_sampler.model.prob(
#    prob_gt_under_model = sp_sampler.conditioned_model.prob(
        (I("BoutAz") < float(groundtruth[0]) + angle_bound) &
        (I("BoutAz") > float(groundtruth[0]) - angle_bound) &
        (I("BoutYaw") < float(groundtruth[1]) + angle_bound) &
        (I("BoutYaw") > float(groundtruth[1]) - angle_bound) &
        (I("BoutDistance") < float(groundtruth[2]) + dist_bound) &
        (I("BoutDistance") > float(groundtruth[2]) - dist_bound))
    if plot_dist:
        posterior_3D_boutplot(posterior_dist, groundtruth)
    return prob_gt_under_model


def get_validation_set(num_bouts_from_end):
    bd = load_bout_data()
    bouts_in_dataset = len(bd["position1"])
    bout_indices = np.arange(bouts_in_dataset - num_bouts_from_end,
                             bouts_in_dataset)
    validation_bout_collector = BoutDataCollector(bout_indices, "validation")
    df = validation_bout_collector.export_bout_dataframe()
    return df


def model_v_groundtruth_probabilities(sp_sampler, *test_df):
    if test_df == ():
        test_df = get_validation_set(3600)
    else:
        test_df = test_df[0]
    probs = []
    for i, df_row in test_df.iterrows():
        p = test_model_v_groundtruth(sp_sampler, df_row, False)
        probs.append(p)
        if i == 100:
            break
    fig, ax = pl.subplots()
    sns.kdeplot(probs, ax=ax, label="probabilities of validation set under model")


def _clustermap(
        D, xticklabels=None, yticklabels=None, vmin=None, vmax=None, **kwargs):
    sns.set_style('white')
    if xticklabels is None:
        xticklabels = range(D.shape[0])
    if yticklabels is None:
        yticklabels = range(D.shape[1])
    zmatrix = sns.clustermap(
        D,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        linewidths=0.2,
        cmap='viridis',
        vmin=vmin,
        vmax=vmax,
    )
    pl.setp(zmatrix.ax_heatmap.get_yticklabels(), rotation=0)
    pl.setp(zmatrix.ax_heatmap.get_xticklabels(), rotation=90)
    return zmatrix


def _clustermap_ordering(D):
    """Returns the ordering of variables in D according to the clustermap."""
    zmatrix = _clustermap(D)
    pl.close(zmatrix.fig)
    xordering = zmatrix.dendrogram_col.reordered_ind
    yordering = zmatrix.dendrogram_row.reordered_ind
    return (xordering, yordering)
    
# if __name__ == "__main__":
#     random_indices = np.cumsum([np.random.randint(200) for i in range(1000)])
#     bd = BoutDataCollector(random_indices)
#     bd.export_bout_dataframe()


'''


tailComponents is arranged by bout, frame, 15 tail segments with an XY angle. only relevant data pieces are tailComponents and deltaPosition

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
