import numpy as np, open3d as o3d
from plyfile import PlyData
ply = PlyData.read("./DATA/equipamento1.ply")
v = ply["vertex"].data
xyz = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float32)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
dists = pcd.compute_nearest_neighbor_distance()
d = np.asarray(dists)
print(f"espaçamento médio: {d.mean():.4f} m")
print(f"mediana:           {np.median(d):.4f} m")
print(f"p95:               {np.percentile(d, 95):.4f} m")
print(f"bbox: {xyz.max(0)-xyz.min(0)}")