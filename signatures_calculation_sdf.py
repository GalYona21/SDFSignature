#!/usr/bin/env python
# coding: utf-8

"""
Script to calculate the mesh curvatures given an implicit representation
of it.
"""

import open3d as o3d
import numpy as np
import torch
import diff_operators
from model import from_pth
from utils import save_ply





def signature_calc(output, H, e1, e2):
    grad_outputs = torch.ones_like(H)
    grad_H = torch.autograd.grad(H, [output["model_in"]], grad_outputs=grad_outputs, create_graph=True)[0]
    H_1 = torch.dot(grad_H, e1)
    H_2 = torch.dot(grad_H, e2)
    grad_H_1 = torch.autograd.grad(H_1, [output["model_in"]], grad_outputs=grad_outputs, create_graph=True)[0]
    H_11 = torch.dot(grad_H_1, e1)
    signature = torch.stack([H, H_1, H_2, H_11])
    return signature



mesh_map = {
    "bunny": ["./data/bunny_curvs.ply", 30],
}



for MESH_TYPE in mesh_map.keys():
    mesh_path, w0 = mesh_map[MESH_TYPE]
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    print(mesh)

    coords = torch.from_numpy(mesh.vertex["positions"].numpy())

    model = from_pth(
        "logs/sdf_bunny/checkpoints/model_current.pth",
        w0=w0
    ).eval()
    print(model)

    out = model(coords.unsqueeze(0))
    X = out['model_in']
    y = out['model_out']

    gradient = diff_operators.gradient(y,X)
    hessian  = diff_operators.hessian(y,X)

    min_curv,max_curv  = diff_operators.principal_curvature(y, X, gradient, hessian[0])
    min_dir ,max_dir   = diff_operators.principal_directions(gradient, hessian[0])

    # mean_curv = diff_operators.mean_curvature(y, X)
    mean_curv = (min_curv+max_curv)*0.5
    signature = signature_calc(out, mean_curv, max_dir, min_dir)

    #vertices of the mesh with the directions of min curvatures and with the min curvatures  (x, v_min, k_min)
    verts_min = np.hstack((coords.squeeze(0).detach().numpy(),
                           min_dir.squeeze(0).detach().numpy(),
                           min_curv.squeeze(0).detach().numpy()))

    #vertices of the mesh with the directions of max curvatures and with the max curvatures  (x, v_max, k_max)
    verts_max = np.hstack((coords.squeeze(0).detach().numpy(),
                           max_dir.squeeze(0).detach().numpy(),
                           max_curv.squeeze(0).detach().numpy()))

    #vertices of the mesh with their normals and with the mean curvatures  (x, N, k_mean)
    verts_mean = np.hstack((coords.squeeze(0).detach().numpy(),
                            mesh.vertex["normals"].numpy(),
                            mean_curv.squeeze(0).detach().numpy()))

    faces = mesh.triangle["indices"].numpy()

    attrs = [("nx", "f4"), ("ny", "f4"), ("nz", "f4"), ("quality", "f4")]
    save_ply(verts_min, faces, f"./results/{MESH_TYPE}_min_curvs.ply",
             vertex_attributes=attrs)
    save_ply(verts_max, faces, f"./results/{MESH_TYPE}_max_curvs.ply",
            vertex_attributes=attrs)
    save_ply(verts_mean, faces, f"./results/{MESH_TYPE}_mean_curvs.ply",
            vertex_attributes=attrs)


