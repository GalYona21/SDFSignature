import numpy as np
import plyfile


def save_ply(
        verts: np.array,
        faces: np.array,
        filename: str,
        vertex_attributes: list = None
) -> None:
    """Converts the vertices and faces into a PLY format, saving the resulting
    file.

    Parameters
    ----------
    verts: np.array
        An NxD matrix with the vertices and its attributes (normals,
        curvatures, etc.). Note that we expect verts to have at least 3
        columns, each corresponding to a vertex coordinate.

    faces: np.array
        An Fx3 matrix with the vertex indices for each triangle.

    filename: str
        Path to the output PLY file.

    vertex_attributes: list of tuples
        A list with the dtypes of vertex attributes other than coordinates.

    Examples
    --------
    > # This creates a simple triangle and saves it to a file called
    > #"triagle.ply"
    > verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    > faces = np.array([[0, 1, 2]])
    > save_ply(verts, faces, "triangle.ply")

    > # Writting normal information as well
    > verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    > faces = np.array([[0, 1, 2]])
    > normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])
    > attrs = [("nx", "f4"), ("ny", "f4"), ("nz", "f4")]
    > save_ply(verts, faces, "triangle_normals.ply", vertex_attributes=attrs)
    """
    # try writing to the ply file
    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    dtypes = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    if vertex_attributes is not None:
        dtypes[3:3] = vertex_attributes

    verts_tuple = np.zeros(
        (num_verts,),
        dtype=dtypes
    )

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(verts[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(
        faces_building,
        dtype=[("vertex_indices", "i4", (3,))]
    )

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    ply_data.write(filename)

