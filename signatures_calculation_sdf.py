#!/usr/bin/env python
# coding: utf-8

"""
Script to calculate the mesh curvatures given an implicit representation
of it.
"""

import numpy as np
import torch
from plyfile import PlyData

import diff_operators
from model import from_pth
from utils import save_ply, rotate_point_cloud


def signature_calc(output, H, e1, e2):
    grad_outputs = torch.ones_like(H)
    grad_H = torch.autograd.grad(H, [output["model_in"]], grad_outputs=grad_outputs, create_graph=True)[0]
    e1 = e1.squeeze(0)
    e2 = e2.squeeze(0)

    H_1 = torch.sum(grad_H * e1, dim=-1).T
    H_2 = torch.sum(grad_H * e2, dim=-1).T
    grad_H_1 = torch.autograd.grad(H_1, [output["model_in"]], grad_outputs=grad_outputs, create_graph=True)[0]
    H_11 = torch.sum(grad_H_1 * e1, dim=-1).T
    signature = torch.stack([H, H_1, H_2, H_11])
    return signature

def load_ply_with_attributes(filename):
    # Load PLY file
    ply_file = PlyData.read(filename)
    vertex_data = ply_file.elements[0]  # properties 'x', 'y', 'z', 'nx', 'ny', 'nz', 'quality'
    normals = np.stack((vertex_data['nx'], vertex_data['ny'], vertex_data['nz']), axis=-1)
    quality = np.array(vertex_data['quality'])
    vertices = np.stack((vertex_data['x'], vertex_data['y'], vertex_data['z']), axis=-1)
    faces = np.vstack(ply_file.elements[1].data['vertex_indices'])

    return vertices, normals, quality, faces

mesh_map = {
    "bunny": ["./data/bunny_curvs.ply", 30],
}



for MESH_TYPE in mesh_map.keys():
    mesh_path, w0 = mesh_map[MESH_TYPE]
    # mesh = o3d.io.read_triangle_mesh(mesh_path)
    # mesh.compute_vertex_normals()
    # mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    # print(mesh)
    xyz, normals, mean_curvature, faces = load_ply_with_attributes(mesh_path)
    xyz, normals = rotate_point_cloud(xyz, normals, angle_x=90, angle_y=30, angle_z=30)

    coords = torch.from_numpy(xyz)
    model_path = "/home/gal.yona/SDFSignatures/SDFSignature/logs/sdf_bunny2/checkpoints/model_current.pth"
    model_path_local = "logs/sdf_rotated_bunny/checkpoints/bunny_model_current.pth"
    model = from_pth(
        model_path,
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
    del gradient
    del hessian
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
                            normals,
                            mean_curv.squeeze(0).detach().numpy()))

    torch.save(signature, f"./results/rotated_{MESH_TYPE}_signature.pt")

    attrs = [("nx", "f4"), ("ny", "f4"), ("nz", "f4"), ("quality", "f4")]
    save_ply(verts_min, faces, f"./results/{MESH_TYPE}_min_curvs.ply",
             vertex_attributes=attrs)
    save_ply(verts_max, faces, f"./results/{MESH_TYPE}_max_curvs.ply",
            vertex_attributes=attrs)
    save_ply(verts_mean, faces, f"./results/{MESH_TYPE}_mean_curvs.ply",
            vertex_attributes=attrs)


