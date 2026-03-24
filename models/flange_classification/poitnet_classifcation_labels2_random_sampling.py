import os
import random
import copy
import numpy as np
import pandas as pd
from plyfile import PlyData

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.neighbors import KDTree
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


# =========================
# CONFIG
# =========================
DATASET_DIR = "../../FLANGE_DATASET_labels2"
CSV_PATH = os.path.join(DATASET_DIR, "flange_info_labels2.csv")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_POINTS = 2048
BATCH_SIZE = 16
NUM_EPOCHS = 120
LEARNING_RATE = 5e-4
PATCHES_PER_EPOCH_TRAIN = 1000
PATCHES_PER_EPOCH_VALID = 200
PATCHES_PER_EPOCH_TEST = 200
SEED = 42

print("Using Device:", DEVICE)


# =========================
# REPRODUTIBILIDADE
# =========================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(SEED)


# =========================
# SPLIT POR EQUIPAMENTO
# =========================
def prepare_dataset():
    df = pd.read_csv(CSV_PATH)

    print("Total de arquivos-base:", len(df))
    print("Equipamentos únicos:", df["Equipamento"].nunique())
    print("\nDistribuição geral:")
    print(df["Label"].value_counts())

    groups = df["Equipamento"]

    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    train_valid_idx, test_idx = next(gss_test.split(df, groups=groups))
    train_valid_df = df.iloc[train_valid_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    gss_valid = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=SEED)
    train_idx_rel, valid_idx_rel = next(
        gss_valid.split(train_valid_df, groups=train_valid_df["Equipamento"])
    )

    train_df = train_valid_df.iloc[train_idx_rel].reset_index(drop=True)
    valid_df = train_valid_df.iloc[valid_idx_rel].reset_index(drop=True)

    print("\nEquipamentos por split")
    print("Train:", train_df["Equipamento"].nunique())
    print("Valid:", valid_df["Equipamento"].nunique())
    print("Test :", test_df["Equipamento"].nunique())

    print("\nClasses no treino:")
    print(train_df["Label"].value_counts())
    print("\nClasses na validação:")
    print(valid_df["Label"].value_counts())
    print("\nClasses no teste:")
    print(test_df["Label"].value_counts())

    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True), test_df.reset_index(drop=True)


# =========================
# LEITURA DOS PLY
# =========================
def read_xyz_from_ply(path):
    plydata = PlyData.read(path)
    vertex = plydata["vertex"].data
    coords = np.column_stack([
        np.array(vertex["x"], dtype=np.float32),
        np.array(vertex["y"], dtype=np.float32),
        np.array(vertex["z"], dtype=np.float32),
    ])
    return coords


def normalize_patch(coords):
    centroid = np.mean(coords, axis=0, keepdims=True)
    coords = coords - centroid
    furthest_distance = np.max(np.sqrt(np.sum(coords ** 2, axis=1)))
    if furthest_distance > 0:
        coords = coords / furthest_distance
    return coords.astype(np.float32)


def augment_patch(coords):
    coords = coords.copy()

    # jitter
    coords += np.random.normal(0, 0.01, size=coords.shape)

    # rotation z
    theta = np.random.uniform(0, 2 * np.pi)
    rot = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ], dtype=np.float32)
    coords = coords @ rot.T

    # scaling
    scale = np.random.uniform(0.9, 1.1)
    coords *= scale

    return coords.astype(np.float32)


# =========================
# DATASET DE PATCHES LOCAIS
# =========================
class LocalPatchDataset(Dataset):
    def __init__(self, df, num_points=2048, epoch_size=1000, train=True):
        self.df = df.reset_index(drop=True)
        self.num_points = num_points
        self.epoch_size = epoch_size
        self.train = train
        self.cache = {}

    def __len__(self):
        return self.epoch_size

    def _load_cloud(self, rel_path):
        full_path = os.path.join(DATASET_DIR, rel_path)

        if full_path not in self.cache:
            coords = read_xyz_from_ply(full_path)
            tree = KDTree(coords)
            self.cache[full_path] = (coords, tree)

        return self.cache[full_path]

    def _sample_local_patch(self, coords, tree, center_idx):
        center = coords[center_idx:center_idx + 1]
        k = min(self.num_points, len(coords))

        _, idx = tree.query(center, k=k)
        patch = coords[idx[0]]

        if len(patch) < self.num_points:
            extra_idx = np.random.choice(len(patch), self.num_points - len(patch), replace=True)
            patch = np.concatenate([patch, patch[extra_idx]], axis=0)

        return patch

    def __getitem__(self, idx):
        if self.train:
            row = self.df.iloc[np.random.randint(0, len(self.df))]
        else:
            row = self.df.iloc[idx % len(self.df)]

        coords, tree = self._load_cloud(row["Arquivo_PLY"])
        label = int(row["Label"])

        if len(coords) == 0:
            raise ValueError(f"Arquivo vazio: {row['Arquivo_PLY']}")

        if self.train:
            center_idx = np.random.randint(0, len(coords))
        else:
            rng = np.random.default_rng(idx + SEED)
            center_idx = int(rng.integers(0, len(coords)))

        patch = self._sample_local_patch(coords, tree, center_idx)
        patch = normalize_patch(patch)

        if self.train:
            patch = augment_patch(patch)

        return torch.from_numpy(patch), label


# =========================
# COLLATE
# =========================
def cloud_collate(batch):
    clouds, labels = list(zip(*batch))
    clouds = torch.stack(clouds, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return clouds, labels


# =========================
# POINTNET++ SIMPLIFICADO
# =========================
def square_distance(src, dst):
    B, N, C = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.transpose(1, 2))
    dist += torch.sum(src ** 2, -1).unsqueeze(-1)
    dist += torch.sum(dst ** 2, -1).unsqueeze(1)
    return dist


def index_points(points, idx):
    B = points.shape[0]
    batch_indices = torch.arange(B, dtype=torch.long).to(points.device).view(B, *((1,) * (idx.dim() - 1)))
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, C = xyz.shape
    npoint = min(npoint, N)
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids


def sample_and_group_all(xyz, points):
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        grouped_points = points.view(B, 1, N, -1)
        new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


def sample_and_group(npoint, radius, nsample, xyz, points):
    B, N, C = xyz.shape
    if npoint is None:
        npoint = N
    else:
        npoint = min(npoint, N)

    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    sqrdists = square_distance(new_xyz, xyz)
    nsample = min(nsample, N)
    group_idx = sqrdists.argsort()[:, :, :nsample]
    grouped_xyz = index_points(xyz, group_idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)

    if points is not None:
        grouped_points = index_points(points, group_idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm

    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample

        last_channel = in_channel
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        if self.npoint is not None:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        else:
            new_xyz, new_points = sample_and_group_all(xyz, points)

        new_points = new_points.permute(0, 3, 1, 2)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, -1)[0]
        new_points = new_points.permute(0, 2, 1)
        return new_xyz, new_points


class PointNet2Classifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=3, mlp=[64, 64, 128])
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024])

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)

        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, xyz):
        points = None
        l1_xyz, l1_points = self.sa1(xyz, points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        _, l3_points = self.sa3(l2_xyz, l2_points)

        x = l3_points.view(xyz.shape[0], -1)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x


# =========================
# TREINO / AVALIAÇÃO
# =========================
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for clouds, labels in tqdm(loader, desc="Train"):
        clouds = clouds.to(device).float()
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(clouds)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, 100.0 * correct / total


def eval_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for clouds, labels in tqdm(loader, desc="Eval"):
            clouds = clouds.to(device).float()
            labels = labels.to(device)

            outputs = model(clouds)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / total, 100.0 * correct / total


def plot_confusion_matrix(model, loader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for clouds, labels in loader:
            clouds = clouds.to(device).float()
            labels = labels.to(device)

            outputs = model(clouds)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot()
    plt.show()
    print(classification_report(all_labels, all_preds, target_names=class_names))


def train_model():
    train_df, valid_df, test_df = prepare_dataset()

    train_dataset = LocalPatchDataset(
        train_df, num_points=NUM_POINTS, epoch_size=PATCHES_PER_EPOCH_TRAIN, train=True
    )
    valid_dataset = LocalPatchDataset(
        valid_df, num_points=NUM_POINTS, epoch_size=PATCHES_PER_EPOCH_VALID, train=False
    )
    test_dataset = LocalPatchDataset(
        test_df, num_points=NUM_POINTS, epoch_size=PATCHES_PER_EPOCH_TEST, train=False
    )

    class_counts = train_df["Label"].value_counts().sort_index()
    class_weights = 1.0 / class_counts
    samples_weight = train_df["Label"].apply(lambda x: class_weights[x]).values
    samples_weight = torch.from_numpy(samples_weight).float()

    # sampler no nível dos arquivos-base
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

    # loader de treino com amostragem normal do dataset virtual
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=cloud_collate
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=cloud_collate
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=cloud_collate
    )

    model = PointNet2Classifier(num_classes=2).to(DEVICE)

    weight_tensor = torch.tensor(
        [1.0, class_counts[0] / class_counts[1]],
        dtype=torch.float32
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_acc = 0.0
    best_model = None

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "valid_loss": [],
        "valid_acc": [],
    }

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        valid_loss, valid_acc = eval_epoch(model, valid_loader, criterion, DEVICE)
        scheduler.step()

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["valid_loss"].append(valid_loss)
        history["valid_acc"].append(valid_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.2f}%")

        if valid_acc > best_acc:
            best_acc = valid_acc
            best_model = copy.deepcopy(model)
            torch.save(best_model.state_dict(), "best_pointnet2_model_random_sampling.pth")
            print("Model saved!")

    pd.DataFrame(history).to_csv("training_history_random_sampling.csv", index=False)

    best_model.load_state_dict(torch.load("best_pointnet2_model_random_sampling.pth", map_location=DEVICE))
    test_loss, test_acc = eval_epoch(best_model, test_loader, criterion, DEVICE)
    print(f"\nTest Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    plot_confusion_matrix(best_model, test_loader, DEVICE, ["Non-Flange", "Flange"])


if __name__ == "__main__":
    train_model()