import json

import numpy as np
from plyfile import PlyData
import open3d as o3d

from utils import rotate_point_cloud, find_vertex_correspondence, find_closest_vertices, unfold_first_signature, \
    unfold_second_signature
import polyscope as ps
import polyscope.imgui as psim


def load_ply_with_attributes(filename):
    # Load PLY file
    ply_file = PlyData.read(filename)
    vertex_data = ply_file.elements[0]  # properties 'x', 'y', 'z', 'nx', 'ny', 'nz', 'quality'
    normals = np.stack((vertex_data['nx'], vertex_data['ny'], vertex_data['nz']), axis=-1)
    quality = np.array(vertex_data['quality'])
    vertices = np.stack((vertex_data['x'], vertex_data['y'], vertex_data['z']), axis=-1)
    faces = np.vstack(ply_file.elements[1].data['vertex_indices'])

    return vertices, normals, quality, faces


def visualize_pointcloud(vertices):
    mesh = o3d.geometry.PointCloud()
    mesh.points = o3d.utility.Vector3dVector(vertices)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])

    o3d.visualization.draw_geometries([mesh, axes])

def visualize_mesh_with_curvatures(xyz, normals, mean_curvature, faces):
    # Create Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(xyz)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    # Visualize with mean curvature as color
    # max_curvature = np.max(mean_curvature)
    # min_curvature = np.min(mean_curvature)
    # normalized_curvature = (mean_curvature - min_curvature) / (max_curvature - min_curvature)

    # Map mean curvature values to colors between blue and red
    # colors = np.zeros((len(normalized_curvature), 3))

    # colors[:, 0] = normalized_curvature  # Red component
    # colors[:, 2] = 1 - normalized_curvature  # Blue component
    # mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    # Visualize with normals
    o3d.visualization.draw_geometries([mesh], point_show_normal=True)


# Load PLY file with attributes
mesh_name = "armadillo"
filename = "/Users/gal.yona/Documents/MSc/ISL Project/SDFSignatures/logs/sdf_meshing/"+mesh_name+".ply"
filename2 = "logs/sdf_meshing/rotated_"+mesh_name+".ply"

# xyz, normals, mean_curvature, faces = load_ply_with_attributes(filename)
# load faces from ply
mesh = o3d.io.read_triangle_mesh(filename)
mesh2 = o3d.io.read_triangle_mesh(filename2)
mesh2.vertices = o3d.utility.Vector3dVector(rotate_point_cloud(np.array(mesh2.vertices),angle_x=90, angle_y=30, angle_z=30, inverse=True)+1)
# mesh2.vertices = o3d.utility.Vector3dVector(np.array(mesh2.vertices)+1)

# vertices, vertex_normals = rotate_point_cloud(np.array(mesh_gt_rotated.vertices), np.array(mesh_gt_rotated.vertex_normals), angle_x=90, angle_y=30, angle_z=30)
# mesh_gt_rotated.vertices = o3d.utility.Vector3dVector(vertices + 1)
# mesh_gt_rotated.vertex_normals = o3d.utility.Vector3dVector(vertex_normals)
# selected_vertices = mesh.vertices[:1000]
#
# # Create a point cloud from the selected vertices
# point_cloud = o3d.geometry.PointCloud()
# point_cloud.points = o3d.utility.Vector3dVector(selected_vertices)
# selected_vertices2 = mesh2.vertices[:1000]

# Create a point cloud from the selected vertices
# point_cloud2 = o3d.geometry.PointCloud()
# point_cloud2.points = o3d.utility.Vector3dVector(selected_vertices2)
#
# selected_vertices_gt = mesh_gt.vertices[:1]
# point_cloud_gt = o3d.geometry.PointCloud()
# point_cloud_gt.points = o3d.utility.Vector3dVector(selected_vertices_gt)
#
# selected_vertices_gt = mesh_gt_rotated.vertices[:1]
# point_cloud_gt_rotated = o3d.geometry.PointCloud()
# point_cloud_gt_rotated.points = o3d.utility.Vector3dVector(selected_vertices_gt)



# Load the JSON file
with open(f"./results/sanity_"+mesh_name+"_signature.json", "r") as file:
    loaded_bunny_signature = json.load(file)
with open(f"./results/sanity_rotated_"+mesh_name+"_signature.json", "r") as file:
    loaded_rotated_bunny_signature = json.load(file)





def plot_mesh_with_vector_fields(mesh, vector_field1, vector_field2):
    vector_field1 = vector_field1.squeeze()
    vector_field2 = vector_field2.squeeze()
    # Initialize Polyscope
    ps.init()

    # Register the mesh
    ps.register_surface_mesh("my mesh", np.array(mesh.vertices), np.array(mesh.triangles), smooth_shade=True)

    # Add the first vector field to the mesh
    ps.get_surface_mesh("my mesh").add_vector_quantity("vector_field1", vector_field1, defined_on='vertices', color=(0.2, 0.5, 0.5))

    # Add the second vector field to the mesh
    ps.get_surface_mesh("my mesh").add_vector_quantity("vector_field2", vector_field2, defined_on='vertices', color=(0.8, 0.2, 0.2))

    # Show the plot
    ps.show()

def plot_meshes_with_fields(meshes, scalar_fields=None, vector_fields=None, point_clouds=None, callback=None):
    # Initialize Polyscope
    ps.init()


    # Register meshes and add fields
    for i, mesh in enumerate(meshes):
        mesh_name = f"mesh_{i}"
        ps.register_surface_mesh(mesh_name, np.array(mesh.vertices), np.array(mesh.triangles), smooth_shade=True)

        if scalar_fields is not None and i < len(scalar_fields):
            for j, scalar_field in enumerate(scalar_fields[i]):
                field_name = f"scalar_field_{i}_{j}"
                ps.get_surface_mesh(mesh_name).add_scalar_quantity(field_name, scalar_field.squeeze(), defined_on='vertices')

        if vector_fields is not None and i < len(vector_fields):
            for j, vector_field in enumerate(vector_fields[i]):
                field_name = f"vector_field_{i}_{j}"
                ps.get_surface_mesh(mesh_name).add_vector_quantity(field_name, vector_field.squeeze(), defined_on='vertices')


        # Register point clouds
    if point_clouds is not None:
        for i, point_cloud in enumerate(point_clouds):
            point_cloud_name = f"point_cloud_{i}"
            ps.register_point_cloud(point_cloud_name, point_cloud)

    if callback is not None:
        ps.set_user_callback(on_frame)
        ps.show()
        ps.clear_user_callback()
    else:
        ps.show()

def plot_mesh_with_scalar_field(mesh, scalar_field):
    scalar_field = scalar_field.squeeze()

    # Initialize Polyscope
    ps.init()

    # Register the mesh
    ps.register_surface_mesh("my mesh", np.array(mesh.vertices), np.array(mesh.triangles), smooth_shade=True)

    # Add the scalar field to the mesh
    ps.get_surface_mesh("my mesh").add_scalar_quantity("scalar_field", scalar_field, defined_on='vertices')

    # Show the plot
    ps.show()

def classify_scalar_field(scalar_field):
    # Compute the 33.3rd and 66.7th percentiles
    percentile_33 = np.percentile(scalar_field, 33.3)
    percentile_67 = np.percentile(scalar_field, 66.7)

    # Classify the scalar field into high, medium, and low categories
    classified_field = np.zeros_like(scalar_field, dtype=int)
    classified_field[scalar_field <= percentile_33] = 0  # Low
    classified_field[(scalar_field > percentile_33) & (scalar_field <= percentile_67)] = 1  # Medium
    classified_field[scalar_field > percentile_67] = 2  # High

    return classified_field


def histogram_equalization(scalar_field, num_bins=256, clip_percentile=0):
    # Compute the percentiles for clipping
    min_val = np.percentile(scalar_field, clip_percentile)
    max_val = np.percentile(scalar_field, 100 - clip_percentile)

    # Clip the scalar field values
    clipped_field = np.clip(scalar_field, min_val, max_val)

    # Compute the histogram of the clipped field
    histogram, bins = np.histogram(clipped_field, bins=num_bins, density=True)

    # Compute the cumulative distribution function (CDF)
    cdf = histogram.cumsum()
    cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min())

    # Map the clipped field values to the equalized values using the CDF
    equalized_field = np.interp(clipped_field, bins[:-1], cdf)

    return equalized_field

# plot_mesh_with_vector_fields(mesh, np.array(loaded_bunny_signature["min_dir"]), np.array(loaded_bunny_signature["max_dir"]))
grad_H = np.array(loaded_bunny_signature["grad_H"])
grad_H2 = np.array(loaded_rotated_bunny_signature["grad_H"])
grad_H = grad_H/np.linalg.norm(grad_H, axis=2)[:,:,None]
grad_H2 = grad_H2/np.linalg.norm(grad_H2, axis=2)[:,:,None]
# plot_mesh_with_vector_fields(mesh,grad_H , np.array(loaded_bunny_signature["max_dir"]))
# plot_mesh_with_vector_fields(mesh2,grad_H2 , np.array(loaded_rotated_bunny_signature["max_dir"]))
# H_classifed = classify_scalar_field(np.array(loaded_bunny_signature["signature"][:1]))
# plot_mesh_with_scalar_field(mesh, H_equalized)

rotated_H, rotated_H_1, rotated_H_2, rotated_H_11 = unfold_first_signature(np.array(loaded_rotated_bunny_signature["signature"]))
rotated_H_equalized = histogram_equalization(rotated_H)
rotated_H_1_equalized = histogram_equalization(rotated_H_1, clip_percentile=1)
rotated_H_2_equalized = histogram_equalization(rotated_H_2)
rotated_H_11_equalized = histogram_equalization(rotated_H_11,clip_percentile=1)


# plot_mesh_with_scalar_field(mesh2, H_equalized)
H, H_1, H_2,H_11 = unfold_first_signature(np.array(loaded_bunny_signature["signature"]))
H_equalized = histogram_equalization(H)
H_1_equalized = histogram_equalization(H_1, clip_percentile=1)
H_2_equalized = histogram_equalization(H_2)
H_11_equalized = histogram_equalization(H_11, clip_percentile=1)
bunny_signature_equalized = [H_equalized,H_1_equalized,H_2_equalized,H_11_equalized]
rotated_bunny_signature_equalized =  [rotated_H_equalized, rotated_H_1_equalized,rotated_H_2_equalized,rotated_H_11_equalized]

plot_meshes_with_fields([mesh, mesh2], scalar_fields=[bunny_signature_equalized, rotated_bunny_signature_equalized],
                        vector_fields=[[grad_H, np.array(loaded_bunny_signature["min_dir"]), np.array(loaded_bunny_signature["max_dir"])],
                                       [grad_H2, np.array(loaded_rotated_bunny_signature["min_dir"]), np.array(loaded_rotated_bunny_signature["max_dir"])]])


bunny_signature_equalized = np.array(bunny_signature_equalized)
rotated_bunny_signature_equalized = np.array(rotated_bunny_signature_equalized)

# correspondence_signature_equalized = find_vertex_correspondence(bunny_signature_equalized, rotated_bunny_signature_equalized)

# mse = (np.linalg.norm(rotated_bunny_signature_equalized - bunny_signature_equalized[:, correspondence_signature_equalized, :]) ** 2) / rotated_bunny_signature_equalized.shape[1]

# plot_meshes_with_fields([mesh, mesh2], point_clouds=[np.array(mesh.vertices)[correspondence_signature_equalized[3000:3005]], np.array(mesh2.vertices)[3000:3005]])

rotated_bunny_signature = np.array(loaded_rotated_bunny_signature["signature"])
bunny_signature = np.array(loaded_bunny_signature["signature"])
# correspondence_signature = find_vertex_correspondence(rotated_bunny_signature, bunny_signature)
#
# point_to_sample = 4000
# plot_meshes_with_fields([mesh, mesh2], point_clouds=[np.array(mesh.vertices)[point_to_sample:point_to_sample+5], np.array(mesh2.vertices)[correspondence_signature[point_to_sample:point_to_sample+5]]])
# print("rotated bunny signature: ")
# print("H: "+str(rotated_bunny_signature[0,point_to_sample:point_to_sample+1,:]))
# print("H_1: "+str(rotated_bunny_signature[1,point_to_sample:point_to_sample+1,:]))
# print("H_2: "+str(rotated_bunny_signature[2,point_to_sample:point_to_sample+1,:]))
# print("H_11: "+str(rotated_bunny_signature[3,point_to_sample:point_to_sample+1,:]))
# print("bunny signature: ")
# print("H: "+str(bunny_signature[0,correspondence_signature[point_to_sample:point_to_sample+1],:]))
# print("H_1: "+str(bunny_signature[1,correspondence_signature[point_to_sample:point_to_sample+1],:]))
# print("H_2: "+str(bunny_signature[2,correspondence_signature[point_to_sample:point_to_sample+1],:]))
# print("H_11: "+str(bunny_signature[3,correspondence_signature[point_to_sample:point_to_sample+1],:]))

# closest_indices = find_closest_vertices(bunny_signature[:,point_to_sample,:], rotated_bunny_signature, 30)

# correspondence_H = find_vertex_correspondence(rotated_bunny_signature[:1,:,:], bunny_signature[:1,:,:])


# test of second signature
bunny_signature2 = np.array(loaded_bunny_signature["signature2"])
rotated_bunny_signature2 = np.array(loaded_rotated_bunny_signature["signature2"])
H,K,H_1,H_2, K_1,K_2 = unfold_second_signature(bunny_signature2)
rotated_H, rotated_K, rotated_H_1, rotated_H_2, rotated_K_1, rotated_K_2 = unfold_second_signature(rotated_bunny_signature2)
H_equalized = histogram_equalization(H)
K_equalized = histogram_equalization(K)
H_1_equalized = histogram_equalization(H_1, clip_percentile=1)
H_2_equalized = histogram_equalization(H_2)
K_1_equalized = histogram_equalization(K_1, clip_percentile=1)
K_2_equalized = histogram_equalization(K_2, clip_percentile=1)
bunny_signature_equalized2 = np.array([H_equalized,K_equalized, H_1_equalized,H_2_equalized, K_1_equalized, K_2_equalized])

rotated_H_equalized = histogram_equalization(rotated_H)
rotated_K_equalized = histogram_equalization(rotated_K)
rotated_H_1_equalized = histogram_equalization(rotated_H_1, clip_percentile=1)
rotated_H_2_equalized = histogram_equalization(rotated_H_2)
rotated_K_1_equalized = histogram_equalization(rotated_K_1, clip_percentile=1)
rotated_K_2_equalized = histogram_equalization(rotated_K_2, clip_percentile=1)
rotated_bunny_signature_equalized2 = np.array([rotated_H_equalized,rotated_K_equalized, rotated_H_1_equalized,rotated_H_2_equalized, rotated_K_1_equalized, rotated_K_2_equalized])

# plot_meshes_with_fields([mesh, mesh2], scalar_fields=[bunny_signature_equalized, rotated_bunny_signature_equalized],
#                         vector_fields=[[grad_H, np.array(loaded_bunny_signature["min_dir"]), np.array(loaded_bunny_signature["max_dir"])],
#                                        [grad_H2, np.array(loaded_rotated_bunny_signature["min_dir"]), np.array(loaded_rotated_bunny_signature["max_dir"])]])


# correspondence_signature2_equalized = find_vertex_correspondence(rotated_bunny_signature_equalized, bunny_signature_equalized)
point_to_sample = 1500

def on_frame():
    global mesh, mesh2, bunny_signature, rotated_bunny_signature, bunny_signature2, rotated_bunny_signature2, bunny_signature_equalized, rotated_bunny_signature_equalized, bunny_signature_equalized2, rotated_bunny_signature_equalized2
    number_of_points_to_vis = 3
    io = psim.GetIO()
    if io.KeyAlt:
        # if ps.has_point_cloud("closest_points"):
        #     ps.remove_point_cloud("closest_points")
        #     ps.remove_point_cloud("closest_points_by_H")
        #     ps.remove_point_cloud("closest_points2")
        #     ps.remove_point_cloud("closest_points_equalized")
        #     ps.remove_point_cloud("closest_points_equalized2")
        screen_coords = io.MousePos
        world_ray = ps.screen_coords_to_world_ray(screen_coords)
        world_pos = ps.screen_coords_to_world_position(screen_coords)
        print(f"Click coords: {screen_coords}")
        print(f" world ray: {world_ray}")
        print(f" world pos: {world_pos}")

        # Find the closest mesh and vertex to the clicked world position
        closest_mesh = None
        closest_vertex_index = None
        closest_distance = float('inf')
        flag= -1
        for i, mesh_obj in enumerate([mesh, mesh2]):
            distances = np.linalg.norm(mesh_obj.vertices - world_pos, axis=1)
            min_distance_index = np.argmin(distances)
            min_distance = distances[min_distance_index]
            if min_distance < closest_distance:
                closest_mesh = mesh_obj
                closest_vertex_index = min_distance_index
                closest_distance = min_distance
                flag = i

        if closest_mesh is not None:
            closest_point = closest_mesh.vertices[closest_vertex_index]
            if flag == 0:
                closest_point_signature = bunny_signature[:,closest_vertex_index,:]
                closest_indices = find_closest_vertices(closest_point_signature, rotated_bunny_signature, s=number_of_points_to_vis)
                closest_indices_by_H = find_closest_vertices(closest_point_signature[:1], rotated_bunny_signature[:1], s=number_of_points_to_vis)
                closest_point_signature2 = bunny_signature2[:, closest_vertex_index, :]
                closest_indices2 = find_closest_vertices(closest_point_signature2, rotated_bunny_signature2, s=number_of_points_to_vis)
                closest_point_signature_equalized = bunny_signature_equalized[:, closest_vertex_index, :]
                closest_indices_equalized = find_closest_vertices(closest_point_signature_equalized, rotated_bunny_signature_equalized, s=number_of_points_to_vis)
                closest_point_signature_equalized2 = bunny_signature_equalized2[:, closest_vertex_index, :]
                closest_indices_equalized2 = find_closest_vertices(closest_point_signature_equalized2,
                                                                  rotated_bunny_signature_equalized2,
                                                                  s=number_of_points_to_vis)
                other_mesh_vertices = mesh2.vertices
            else:
                closest_point_signature = rotated_bunny_signature[:, closest_vertex_index, :]
                closest_indices = find_closest_vertices(closest_point_signature, bunny_signature, s=number_of_points_to_vis)
                closest_indices_by_H = find_closest_vertices(closest_point_signature[:1], bunny_signature[:1], s=number_of_points_to_vis)
                closest_point_signature2 = rotated_bunny_signature2[:, closest_vertex_index, :]
                closest_indices2 = find_closest_vertices(closest_point_signature2, bunny_signature2, s=number_of_points_to_vis)
                closest_point_signature_equalized = rotated_bunny_signature_equalized[:, closest_vertex_index, :]
                closest_indices_equalized = find_closest_vertices(closest_point_signature_equalized, bunny_signature_equalized,
                                                         s=number_of_points_to_vis)
                closest_point_signature_equalized2 = rotated_bunny_signature_equalized2[:, closest_vertex_index, :]
                closest_indices_equalized2 = find_closest_vertices(closest_point_signature_equalized2,
                                                                  bunny_signature_equalized2,
                                                                  s=number_of_points_to_vis)
                other_mesh_vertices = mesh.vertices


            # print(f"Closest points to {closest_point}: {closest_indices}")
            # Create a list of points representing the closest point and the 10 closest points on the other mesh
            points = np.array([closest_point] + [other_mesh_vertices[idx] for idx in closest_indices])
            points_by_H = np.array([other_mesh_vertices[idx] for idx in closest_indices_by_H])
            points2 = np.array([other_mesh_vertices[idx] for idx in closest_indices2])
            points_equalized = np.array([other_mesh_vertices[idx] for idx in closest_indices_equalized])
            points_equalized2 = np.array([other_mesh_vertices[idx] for idx in closest_indices_equalized2])
            # Create a list of colors for the points (optional)
            # colors = np.array([[1.0, 0.0, 0.0]] + [[0.0, 1.0, 0.0] for _ in range(len(closest_indices))])

            # Register the point cloud with Polyscope
            ps.register_point_cloud("closest_points", points, color=[1,0,0])
            ps.register_point_cloud("closest_points2", points2,  color=[0,0,1])
            ps.register_point_cloud("closest_points_equalized", points_equalized,  color=[0,1,1])
            ps.register_point_cloud("closest_points_equalized2", points_equalized,  color=[1,1,0])
            ps.register_point_cloud("closest_points_by_H", points_by_H, color=[0,1,0])




# plot_meshes_with_fields([mesh, mesh2], point_clouds=[np.array(mesh.vertices)[point_to_sample:point_to_sample+5], np.array(mesh2.vertices)[correspondence_signature2_equalized[point_to_sample:point_to_sample+5]]], callback=on_frame)
plot_meshes_with_fields([mesh, mesh2], callback=on_frame)
