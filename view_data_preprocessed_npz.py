import numpy as np
import open3d as o3d

path = r"./DATA_PREPROCESSED_v5/equipamento10.npz"
data = np.load(path)

features = data["features"]
labels = data["labels"]

xyz = features[:, :3]   # xyz normalizado
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))

colors = np.zeros((len(labels), 3), dtype=np.float64)
colors[labels == 0] = [0.7, 0.7, 0.7]  # equipamento = cinza
colors[labels == 1] = [1.0, 0.0, 0.0]  # flange = vermelho

pcd.colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries([pcd], window_name="NPZ visualizado")