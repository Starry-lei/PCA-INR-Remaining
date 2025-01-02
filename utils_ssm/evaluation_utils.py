import os
from typing import List, Tuple
import numpy as np
import trimesh
import open3d as o3d



def save_point_cloud(point_cloud, filename):
    """
    Save a 3D point cloud to a file using Open3D.
    Args:
        point_cloud: NumPy array with shape (N, 3) representing 3D coordinates of points.
        filename:    Name of the file to save (e.g., "output.ply").
    """
    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    # Save the point cloud to the specified file
    o3d.io.write_point_cloud(filename, pcd)


def visualize_point_cloud(point_cloud):
    """
    Visualize a 3D point cloud using Open3D.
    Args:
        point_cloud: NumPy array with shape (N, 3) representing 3D coordinates of points.
    """
    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])


def get_correspondended_vertices(
    idx_train_shapes: List[int], path: str
) -> Tuple[np.ndarray, int]:
    """
    Get corresponded vertices from predictions
    Args:
        idx_train_shapes:               Indices of shapes used to build the SSM
        path:                           Path to saved npy file with correspondences
    Return:
        corresponded_vertices_train:    Corresponded vertices of specific shapes
        n_points:                       Number of points in SSM (flattened)
    """
    demo = np.load(path)
    # print("shape of demo",demo.shape)
    corresponded_vertices_all = np.transpose(np.load(path), (0, 2, 1))

    # print("corresponded_vertices_all shape:", corresponded_vertices_all.shape)

    n_shapes = corresponded_vertices_all.shape[2]
    # print("n_shapes:", n_shapes)
    corresponded_vertices_all = corresponded_vertices_all.reshape(-1, n_shapes)
    # print("corresponded_vertices_all shape after reshape:", corresponded_vertices_all.shape)

    # corresponded_vertices_train = corresponded_vertices_all[:, idx_train_shapes]
    corresponded_vertices_train = corresponded_vertices_all[:, :]

    n_points = corresponded_vertices_train.shape[0]

    # print("check corresponded_vertices_train shape:", corresponded_vertices_train.shape)
    # print("check n_points:", n_points)

    return corresponded_vertices_train, n_points


def get_target_point_cloud(path: str, val_idx: int) -> trimesh.Trimesh:
    """
    Get original target point cloud to measure error.
    Args:
        path:       Path to directory which contain the original meshes
        val_idx:    Index of target shape
    Returns:
        target:     Centered target point cloud
    """
    target = trimesh.load(os.path.join(path, "{:03d}.ply".format(val_idx))).vertices
    target -= np.mean(target, axis=0)
    return target


def get_test_point_cloud(path: str, val_idx: int) -> np.ndarray:
    """
    Get test point cloud to be reconstructed by the SSM.
    Already in correspondence with other shapes.
    Args:
        path:               Path to saved npy file with correspondences
        val_idx:            Index of test shape
    Returns:
        test_point_cloud:   Flattened point cloud in correspondence
    """
    corresponded_vertices_all = np.transpose(np.load(path), (0, 2, 1))
    test_point_cloud = corresponded_vertices_all[..., val_idx]
    # print("test_point_cloud points number:",len(test_point_cloud))
    # visualize_point_cloud(test_point_cloud)
    # save_point_cloud(test_point_cloud,"test_point_cloud.pcd")
    return test_point_cloud.reshape(-1)
