import argparse
import os
import numpy as np
import scipy.spatial as spatial
from tqdm import tqdm
import plotly.graph_objects as go


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, default='./pclouds', help='input folder (point clouds)')
    parser.add_argument('--filename', type=str, default='noise_free_and_noise.txt',
                        help='output folder (trained models)')
    parser.add_argument('--logdir', type=str, default='./logs', help='training log folder')
    parser.add_argument('--html_dir', type=str, default='./htmlfiles', help='htmlfiles folder')
    parser.add_argument('--num_neighbours', type=int, default=10, help='num_neighbours')
    parser.add_argument('--points_per_patch', type=int, default=500, help='max. number of points per patch')

    return parser.parse_args()


def compute_ang_err(n0, n):
    n0 = np.broadcast_to(np.expand_dims(n0, axis=0), (len(n), 3))
    err = np.minimum(((n0 - n) ** 2).sum(1), ((n0 + n) ** 2).sum(1))
    return err


def visualise(opt, plot_index, neigh_pts, neigh_normals, pts, normal0):
    arrow_tip_ratio = 0.2
    arrow_starting_ratio = 0.98
    normal0 = normal0 / 5
    neigh_normals = neigh_normals / 5
    if not os.path.exists(opt.html_dir):
        os.mkdir(opt.html_dir)
    html_path = os.path.join(opt.html_dir, "%d.html" % plot_index)
    layout = go.Layout(title='3D Gaussian Normal Visualization',
                       scene=dict(aspectmode='cube',  # this string can be 'data', 'cube', 'auto', 'manual'
                                  # a custom aspectratio is defined as follows:
                                  xaxis=dict(
                                      # range=[-1, 1],
                                      showgrid=False,  # thin lines in the background
                                      zeroline=False,  # thick line at x=0
                                      visible=False),
                                  yaxis=dict(
                                      # range=[-1, 1],
                                      showgrid=False,  # thin lines in the background
                                      zeroline=False,  # thick line at x=0
                                      visible=False
                                  ),
                                  zaxis=dict(
                                      # range=[-1, 1],
                                      showgrid=False,  # thin lines in the background
                                      zeroline=False,  # thick line at x=0
                                      visible=False
                                  ),
                                  aspectratio=dict(x=1, y=1, z=1)
                                  )
                       )
    fig = go.Figure(layout=layout)
    # for i, point in enumerate(pts):
    #     fig.add_trace(go.Scatter3d(
    #         x=[point[0], point[0] + normal[i, 0] / 2],
    #         y=[point[1], point[1] + normal[i, 1] / 2],
    #         z=[point[2], point[2] + normal[i, 2] / 2],
    #         mode='lines',
    #         showlegend=False,
    #         line=dict(width=2, color='rgb(0, 255, 0)')
    #     ))
    for i, point in enumerate(neigh_pts):
        fig.add_trace(go.Scatter3d(
            x=[point[0], point[0] + neigh_normals[i, 0] / 2],
            y=[point[1], point[1] + neigh_normals[i, 1] / 2],
            z=[point[2], point[2] + neigh_normals[i, 2] / 2],
            mode='lines',
            showlegend=False,
            line=dict(width=2, color='rgb(0, 255, 0)')
        ))
        fig.add_trace(go.Cone(
            x=[point[0] + arrow_starting_ratio * neigh_normals[i, 0] / 2],
            y=[point[1] + arrow_starting_ratio * neigh_normals[i, 1] / 2],
            z=[point[2] + arrow_starting_ratio * neigh_normals[i, 2] / 2],
            u=[point[0] + arrow_tip_ratio * neigh_normals[i, 0] / 2],
            v=[point[1] + arrow_tip_ratio * neigh_normals[i, 1] / 2],
            w=[point[2] + arrow_tip_ratio * neigh_normals[i, 2] / 2],
            showlegend=False,
            showscale=False,
            colorscale=[[0, 'rgb(255,0,0)'], [1, 'rgb(255,0,0)']]
        ))
    fig.add_trace(go.Scatter3d(
        x=neigh_pts[:, 0],
        y=neigh_pts[:, 1],
        z=neigh_pts[:, 2],
        mode='markers',
        showlegend=False,
        marker=dict(
            size=1,
            color='blue',
        ),
    ))
    fig.add_trace(go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode='markers',
        showlegend=False,
        marker=dict(
            size=1,
            color='blue',
        ),
    ))
    fig.add_trace(go.Scatter3d(
        x=np.array([0]),
        y=np.array([0]),
        z=np.array([0]),
        mode='markers',
        showlegend=False,
        marker=dict(
            size=1,
            color='red',
        ),
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, normal0[0]],
        y=[0, normal0[1]],
        z=[0, normal0[2]],
        mode='lines',
        showlegend=False,
        line=dict(width=2, color='black')
    ))

    fig.add_trace(go.Cone(
        x=[arrow_starting_ratio * normal0[0]],
        y=[arrow_starting_ratio * normal0[1]],
        z=[arrow_starting_ratio * normal0[2]],
        u=[arrow_tip_ratio * normal0[0]],
        v=[arrow_tip_ratio * normal0[1]],
        w=[arrow_tip_ratio * normal0[2]],
        showlegend=False,
        showscale=False,
        colorscale=[[0, 'rgb(255,0,0)'], [1, 'rgb(255,0,0)']]
    ))
    # fig.show()
    fig.write_html(html_path)
    print('html file has been saved to %s ' % html_path)


class PointCloud:
    def __init__(self, opt, type='noise_free'):
        self.root = opt.indir
        self.shape_names = []
        self.patch_size = opt.points_per_patch
        self.k = opt.num_neighbours
        self.seed = np.random.randint(0, 2 ** 32)
        self.rng = np.random.RandomState(self.seed)
        with open(os.path.join(self.root, opt.filename)) as f:
            self.shape_names = f.readlines()
        self.shape_names = [x.strip() for x in self.shape_names]
        self.shape_names = list(filter(None, self.shape_names))
        if type == 'noise_free':
            point_filename = os.path.join(self.root, self.shape_names[0] + '.xyz')
            normals_filename = os.path.join(self.root, self.shape_names[0] + '.normals')
        elif type == 'noise':
            point_filename = os.path.join(self.root, self.shape_names[1] + '.xyz')
            normals_filename = os.path.join(self.root, self.shape_names[1] + '.normals')
        else:
            raise ValueError('Unknown type of PointCloud: noise_free ot noise ?')

        self.pts = np.loadtxt(point_filename)
        self.normals = np.loadtxt(normals_filename)

    def __len__(self):
        return len(self.pts)

    def compute_score(self, noise_free_pcloud):
        print("Computing the scores !".center(50, "-"))
        iim = np.empty((0, self.k))
        ssm = []
        noise_free_pcloud.nftree = spatial.cKDTree(noise_free_pcloud.pts, 500)
        for j, point in enumerate(tqdm(self.pts)):
            dd, ii = noise_free_pcloud.nftree.query(point, k=self.k)
            iim = np.append(iim, np.expand_dims(ii, axis=0), axis=0)
            ang_err = compute_ang_err(self.normals[j], noise_free_pcloud.normals[ii])
            score = np.sum(ang_err / dd)
            ssm.append(score)
        return np.array(ssm), iim

    def get_patch(self, center_point):
        # _, patch_inds = self.nftree.query(center_point, self.patch_size)
        patch_inds = self.rng.choice(self.__len__(), int(0.1 * self.__len__()), replace=False)
        patch_normals = self.normals[patch_inds]
        patch_pts = self.pts[patch_inds] - center_point
        patch_rad = float(np.linalg.norm(patch_pts.max(0) - patch_pts.min(0), 2))
        patch_pts = patch_pts / patch_rad
        return patch_pts, patch_normals, patch_rad


if __name__ == '__main__':
    opt = parse_arguments()
    nfp = PointCloud(opt, type='noise_free')
    noisep = PointCloud(opt, type='noise')
    score, iim = noisep.compute_score(nfp)
    sort_index = np.argsort(score)[::-1]
    plot_index = int(input('Please enter a number you wanna plot from 1 to %d :' % len(score)))
    patch_pts, patch_normals, rad = nfp.get_patch(noisep.pts[sort_index[plot_index - 1]])
    neigh_pts = (nfp.pts[iim[sort_index[plot_index - 1]].astype(int)] - noisep.pts[sort_index[plot_index - 1]]) / rad
    neigh_normals = nfp.normals[iim[sort_index[plot_index - 1]].astype(int)]
    visualise(opt, plot_index, neigh_pts, neigh_normals, patch_pts, noisep.normals[sort_index[plot_index - 1]])
