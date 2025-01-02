
import open3d as o3d
import numpy as np

def getKeypoints(mesh_roar, mesh_keypointNet, label: dict):
    keypoints = []
    bbox1 = mesh_roar.get_axis_aligned_bounding_box()
    bbox2 = mesh_keypointNet.get_axis_aligned_bounding_box()
    translation = bbox1.get_center() - bbox2.get_center()
    # align bounding box
    mesh_keypointNet.translate(translation)

    # scale bounding box
    extents1 = bbox1.get_extent()
    extents2 = bbox2.get_extent()
    scale = extents1 / extents2

    vertices = np.array(mesh_keypointNet.vertices)
    faces = np.array(mesh_keypointNet.triangles)

    for kp in label['keypoints']:
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        face_coords = vertices[faces[kp['mesh_info']['face_index']]]
        mesh_sphere.translate(face_coords.T @ kp['mesh_info']['face_uv'])

        mesh_sphere.scale(scale.max(), center = mesh_keypointNet.get_axis_aligned_bounding_box().get_center())
        keypoints.append(mesh_sphere.get_center())

    return keypoints

