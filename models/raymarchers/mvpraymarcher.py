# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
""" Raymarcher for a mixture of volumetric primitives """
import os
import itertools
import time
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import plotly.graph_objs as go
from plotly.graph_objects import Figure, Scatter3d

from extensions.mvpraymarch.mvpraymarch import mvpraymarch


#################################################################################
#              Visualization Functions for 3D Data and Trajectories             #
#################################################################################


def visualize_point_clouds(
    pcd, color="plum", size=10, fig=None, show=False, path="pcs.html", save=False
):
    # color scheme: https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter3d.html#plotly.graph_objects.Scatter3d
    assert path is not None if save else True
    # pcd: (N, 3)
    if fig is None:
        fig = Figure()
    fig.add_trace(
        Scatter3d(
            x=pcd[:, 0],
            y=pcd[:, 1],
            z=pcd[:, 2],
            mode="markers",
            marker=dict(size=size, color=color, opacity=0.8),
        )
    )
    if show:
        fig.show()
    if save:
        fig.write_html(path)
    return fig

def visualize_3d_vectors_at_position(
    position,
    vectors,
    labels=None,
    colors=None,
    arrowhead_size=0.1,
    fig=None,
    show=True,
    xyz_limits=None,
    vector_len=1.0,
):
    # controlling the size of the vectors
    vectors = vector_len * vectors

    # vectors: shape [N, 3]
    if colors is None:
        colors = ["blue"] * len(vectors)

    if labels is None:
        labels = [f"Vector {i}" for i in range(len(vectors))]

    if fig is None:
        fig = go.Figure()

    for vec, label, color in zip(vectors, labels, colors):
        x0, y0, z0 = position
        x1, y1, z1 = x0 + vec[0], y0 + vec[1], z0 + vec[2]

        # Arrow shaft (line segment)
        fig.add_trace(
            go.Scatter3d(
                x=[x0, x1],
                y=[y0, y1],
                z=[z0, z1],
                mode="lines",
                line=dict(color=color, width=4),
                showlegend=False,
                hoverinfo="none",
            )
        )

        # Arrowhead (cone)
        fig.add_trace(
            go.Cone(
                x=[x1],
                y=[y1],
                z=[z1],
                u=[vec[0]],
                v=[vec[1]],
                w=[vec[2]],
                hovertext=[label],
                showlegend=True,
                showscale=False,
                sizemode="absolute",
                sizeref=arrowhead_size,
                anchor="tail",
                colorscale=[[0, color], [1, color]],
                name=label,
            )
        )

    if show:
        fig.show()
    return fig

def pytorch_mvpraymarch(
    raypos, raydir, stepsize, tminmax, prims, template, warp=None, rayterm=None,
    fadeexp=8.0, fadescale=8.0, accum=0, dowarp=False,
):
    # python raymarching implementation
    import torch
    primpos, primrot, primscale = prims[0], prims[1], prims[2]
    K = primpos.shape[1]
    N = raypos.shape[0]
    H = raypos.shape[2]
    W = raypos.shape[3]

    rayrgba = torch.zeros((N, H, W, 4)).to("cuda")
    raypos = raypos + raydir * tminmax[:, :, :, 0, None]
    t = tminmax[:, :, :, 0]

    step = 0
    t0 = t.detach().clone()
    raypos0 = raypos.detach().clone()

    torch.cuda.synchronize()

    while (t < tminmax[:, :, :, 1]).any():
        for k in range(K):
            y0 = torch.bmm(
                    (raypos - primpos[:, k, None, None, :]).view(raypos.size(0), -1, raypos.size(3)),
                    primrot[:, k, :, :]).view_as(raypos) * primscale[:, k, None, None, :]

            fade = torch.exp(-fadescale * torch.sum(torch.abs(y0) ** fadeexp, dim=-1, keepdim=True))

            if dowarp:
                y1 = F.grid_sample(
                        warp[:, k, :, :, :, :],
                        y0[:, None, :, :, :], align_corners=True)[:, :, 0, :, :].permute(0, 2, 3, 1)
            else:
                y1 = y0

            sample = F.grid_sample(
                    template[:, k, :, :, :, :],
                    y1[:, None, :, :, :], align_corners=True)[:, :, 0, :, :].permute(0, 2, 3, 1)

            valid1 = (
                torch.prod(y0[:, :, :, :] >= -1., dim=-1, keepdim=True) *
                torch.prod(y0[:, :, :, :] <= 1., dim=-1, keepdim=True))

            valid = ((t >= tminmax[:, :, :, 0]) & (t < tminmax[:, :, :, 1])).float()[:, :, :, None]

            alpha0 = sample[:, :, :, 3:4]

            rgb = sample[:, :, :, 0:3] * valid * valid1
            alpha = alpha0 * fade * stepsize * valid * valid1

            if accum == 0:
                newalpha = rayrgba[:, :, :, 3:4] + alpha
                contrib = (newalpha.clamp(max=1.0) - rayrgba[:, :, :, 3:4]) * valid * valid1
                rayrgba = rayrgba + contrib * torch.cat([rgb, torch.ones_like(alpha)], dim=-1)
            else:
                raise

        step += 1
        t = t0 + stepsize * step
        raypos = raypos0 + raydir * stepsize * step

    print(rayrgba[..., -1].min().item(), rayrgba[..., -1].max().item())

    return rayrgba


class Raymarcher(nn.Module):
    def __init__(self, volradius):
        super(Raymarcher, self).__init__()

        self.volradius = volradius

    def forward(self, raypos, raydir, tminmax, decout,
            encoding=None, renderoptions={}, trainiter=-1, evaliter=-1,
            rayterm=None,
            **kwargs):

        # rescale world
        dt = renderoptions["dt"] / self.volradius
        # rayrgba_pytorch = pytorch_mvpraymarch(raypos, raydir, dt, tminmax,
        #         (decout["primpos"], decout["primrot"], decout["primscale"]),
        #         template=decout["template"],
        #         warp=decout["warp"] if "warp" in decout else None,
        #         rayterm=rayterm,
        #         )
                # **{k:v for k, v in renderoptions.items() if k in mvpraymarch.__code__.co_varnames})

        rayrgba = mvpraymarch(raypos, raydir, dt, tminmax,
                (decout["primpos"], decout["primrot"], decout["primscale"]),
                template=decout["template"],
                warp=decout["warp"] if "warp" in decout else None,
                rayterm=rayterm,
                **{k:v for k, v in renderoptions.items() if k in mvpraymarch.__code__.co_varnames})
        
        
        # one_position = raypos[0, 512:514, 512, :].cpu().numpy()
        # one_direction = raydir[0, 512:514, 512, :].cpu().numpy()
        # one_tminmax = tminmax[0, 512:514, 512, :].cpu().numpy()

        # primpos = decout["primpos"][0, :, :].cpu().detach().numpy()

        # fig = visualize_point_clouds(primpos, color="plum", size=10, fig=None, show=True, path="pcs.html", save=False)
        # fig = visualize_3d_vectors_at_position(one_position[0], one_direction[0].reshape(-1, 3), labels=[""], arrowhead_size=50, fig=fig, vector_len=100.0)

        # fig.add_trace(
        #     Scatter3d(
        #         x=np.array(one_position[0]),
        #         y=np.array(one_position[1]),
        #         z=np.array(one_position[2]),
        #         mode="markers",
        #         marker=dict(size=10, color="aliceblue", opacity=0.8),
        #     )
        # )


        # fig.write_html("test.html")

        return rayrgba.permute(0, 3, 1, 2), {}
