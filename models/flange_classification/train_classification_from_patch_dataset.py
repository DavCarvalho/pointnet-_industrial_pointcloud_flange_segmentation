import os
import copy
import random
import argparse
import numpy as np
import pandas as pd
from plyfile import PlyData

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# REPRODUTIBILIDADE
# ============================================================

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# LEITURA / NORMALIZAÇÃO
# ============================================================

def read_xyz_from_ply(path: str) -> np.ndarray:
    plydata = PlyData.read(path)
    vertex = plydata["vertex"].data
    xyz = np.column_stack([
        np.asarray(vertex["x"], dtype=np.float32),
        np.asarray(vertex["y"], dtype=np.float32),
        np.asarray(vertex["z"], dtype=np.float32),
    ])
    return xyz


def normalize_patch(coords: np.ndarray) -> np.ndarray:
    coords = coords.astype(np.float32)
    centroid = np.mean(coords, axis=0, keepdims=True)
    coords = coords - centroid
    furthest_distance = np.max(np.sqrt(np.sum(coords ** 2, axis=1)))
    if furthest_distance > 1e-8:
        coords = coords / furthest_distance
    return coords.astype(np.float32)


def augment_patch(coords: np.ndarray) -> np.ndarray:
    coords = coords.copy().astype(np.float32)

    # jitter
    coords += np.random.normal(0, 0.01, size=coords.shape).astype(np.float32)

    # rotação no eixo z
    theta = np.random.uniform(0, 2 * np.pi)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1],
    ], dtype=np.float32)
    coords = coords @ rot.T

    # escala
    scale = np.random.uniform(0.9, 1.1)
    coords *= scale

    return coords.astype(np.float32)


def resample_to_num_points(coords: np.ndarray, num_points: int, seed: int = None) -> np.ndarray:
    n = len(coords)
    if n == 0:
        raise ValueError("Patch vazio.")

    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)

    if n == num_points:
        return coords
    elif n > num_points:
        idx = rng.choice(n, size=num_points, replace=False)
    else:
        extra = rng.choice(n, size=(num_points - n), replace=True)
        idx = np.concatenate([np.arange(n), extra])
        rng.shuffle(idx)
    return coords[idx]


# ============================================================
# DATASET
# ============================================================

class PatchClassificationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, dataset_dir: str, num_points: int = 2048,
                 train: bool = True, augment: bool = True):
        self.df = df.reset_index(drop=True)
        self.dataset_dir = dataset_dir
        self.num_points = num_points
        self.train = train
        self.augment = augment
        self.cache = {}

    def __len__(self):
        return len(self.df)

    def _load_patch(self, rel_path: str) -> np.ndarray:
        full_path = os.path.join(self.dataset_dir, rel_path)
        if full_path not in self.cache:
            self.cache[full_path] = read_xyz_from_ply(full_path)
        return self.cache[full_path]

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        coords = self._load_patch(row["Arquivo_PLY"])
        label = int(row["Label"])

        if self.train:
            coords = resample_to_num_points(coords, self.num_points, seed=None)
        else:
            coords = resample_to_num_points(coords, self.num_points, seed=idx + 12345)

        coords = normalize_patch(coords)

        if self.train and self.augment:
            coords = augment_patch(coords)

        return torch.from_numpy(coords.astype(np.float32)), torch.tensor(label, dtype=torch.long)


# ============================================================
# POINTNET++ SIMPLIFICADO
# ============================================================

def square_distance(src, dst):
    dist = -2 * torch.matmul(src, dst.transpose(1, 2))
    dist += torch.sum(src ** 2, dim=-1).unsqueeze(-1)
    dist += torch.sum(dst ** 2, dim=-1).unsqueeze(1)
    return dist


def index_points(points, idx):
    B = points.shape[0]
    batch_indices = torch.arange(B, dtype=torch.long, device=points.device).view(
        B, *((1,) * (idx.dim() - 1))
    )
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, C = xyz.shape
    npoint = min(npoint, N)

    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]

    return centroids


def sample_and_group_all(xyz, points):
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C, device=device)
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
            new_xyz, new_points = sample_and_group(
                self.npoint, self.radius, self.nsample, xyz, points
            )
        else:
            new_xyz, new_points = sample_and_group_all(xyz, points)

        new_points = new_points.permute(0, 3, 1, 2)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, dim=-1)[0]
        new_points = new_points.permute(0, 2, 1)
        return new_xyz, new_points


class PointNet2Classifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32, in_channel=3, mlp=[64, 64, 128]
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256]
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024]
        )

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

        x = l3_points.reshape(xyz.shape[0], -1)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x


# ============================================================
# SPLIT
# ============================================================

def split_by_equipment(df: pd.DataFrame, seed: int):
    groups = df["Equipamento"]

    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_valid_idx, test_idx = next(gss_test.split(df, groups=groups))
    train_valid_df = df.iloc[train_valid_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    gss_valid = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=seed)
    train_idx_rel, valid_idx_rel = next(
        gss_valid.split(train_valid_df, groups=train_valid_df["Equipamento"])
    )

    train_df = train_valid_df.iloc[train_idx_rel].reset_index(drop=True)
    valid_df = train_valid_df.iloc[valid_idx_rel].reset_index(drop=True)
    return train_df, valid_df, test_df


# ============================================================
# MÉTRICAS
# ============================================================

def evaluate_predictions(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p_bin, r_bin, f1_bin, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0, pos_label=1
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": acc,
        "precision_flange": p_bin,
        "recall_flange": r_bin,
        "f1_flange": f1_bin,
        "macro_precision": p_macro,
        "macro_recall": r_macro,
        "macro_f1": f1_macro,
        "cm": cm,
    }


# ============================================================
# TREINO / EVAL
# ============================================================

def run_epoch(model, loader, device, criterion, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss = 0.0
    all_true = []
    all_pred = []

    with torch.set_grad_enabled(train_mode):
        for clouds, labels in loader:
            clouds = clouds.to(device).float()
            labels = labels.to(device)

            if train_mode:
                optimizer.zero_grad()

            outputs = model(clouds)
            loss = criterion(outputs, labels)

            if train_mode:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            all_true.extend(labels.detach().cpu().numpy().tolist())
            all_pred.extend(preds.detach().cpu().numpy().tolist())

    metrics = evaluate_predictions(all_true, all_pred)
    metrics["loss"] = total_loss / max(len(all_true), 1)
    return metrics, np.array(all_true), np.array(all_pred)


# ============================================================
# PLOTS / RELATÓRIO
# ============================================================

def save_curves(history_df: pd.DataFrame, out_path: str):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history_df["epoch"], history_df["train_loss"], label="Train")
    plt.plot(history_df["epoch"], history_df["valid_loss"], label="Valid")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history_df["epoch"], history_df["train_macro_f1"], label="Train Macro-F1")
    plt.plot(history_df["epoch"], history_df["valid_macro_f1"], label="Valid Macro-F1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("Macro-F1")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def save_confusion_matrix(cm: np.ndarray, out_path: str):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["non_flange", "flange"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, colorbar=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Treina classificador PointNet++ no dataset de patches já construídos.")
    parser.add_argument("--dataset_dir", type=str, default="../../build_datasets_flanges/FLANGE_CLASSIFICATION_PATCH_DATASET")
    parser.add_argument("--csv_path", type=str, default=None)
    parser.add_argument("--num_points", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--early_stop_patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="./classification_patch_training_outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_path = args.csv_path or os.path.join(args.dataset_dir, "classification_patch_info.csv")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Device: {device}")
    print(f"Dataset dir: {args.dataset_dir}")
    print(f"CSV path: {csv_path}")
    print(f"Augmentation: {args.augment}")

    df = pd.read_csv(csv_path)
    if not {"Equipamento", "Arquivo_PLY", "Label"}.issubset(df.columns):
        raise ValueError("O CSV precisa ter pelo menos as colunas: Equipamento, Arquivo_PLY, Label")

    train_df, valid_df, test_df = split_by_equipment(df, seed=args.seed)

    print("\nTotal de patches:", len(df))
    print("Equipamentos únicos:", df["Equipamento"].nunique())
    print("\nSplit por equipamento:")
    print(f"Train: {train_df['Equipamento'].nunique()} equipamentos | {len(train_df)} patches")
    print(f"Valid: {valid_df['Equipamento'].nunique()} equipamentos | {len(valid_df)} patches")
    print(f"Test : {test_df['Equipamento'].nunique()} equipamentos | {len(test_df)} patches")
    print("\nDistribuição de classes:")
    print("Train:\n", train_df["Label"].value_counts().sort_index())
    print("Valid:\n", valid_df["Label"].value_counts().sort_index())
    print("Test:\n", test_df["Label"].value_counts().sort_index())

    train_dataset = PatchClassificationDataset(
        train_df, dataset_dir=args.dataset_dir, num_points=args.num_points, train=True, augment=args.augment
    )
    valid_dataset = PatchClassificationDataset(
        valid_df, dataset_dir=args.dataset_dir, num_points=args.num_points, train=False, augment=False
    )
    test_dataset = PatchClassificationDataset(
        test_df, dataset_dir=args.dataset_dir, num_points=args.num_points, train=False, augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = PointNet2Classifier(num_classes=2).to(device)

    class_counts = train_df["Label"].value_counts().sort_index()
    n0 = int(class_counts.get(0, 1))
    n1 = int(class_counts.get(1, 1))
    class_weights = torch.tensor([
        1.0,
        n0 / max(n1, 1),
    ], dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    history = []
    best_metric = -1.0
    best_epoch = -1
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0

    model_path = os.path.join(args.out_dir, f"best_model_seed{args.seed}.pth")
    history_path = os.path.join(args.out_dir, f"history_seed{args.seed}.csv")
    curves_path = os.path.join(args.out_dir, f"curves_seed{args.seed}.png")
    cm_path = os.path.join(args.out_dir, f"test_cm_seed{args.seed}.png")
    report_path = os.path.join(args.out_dir, f"test_report_seed{args.seed}.txt")
    test_metrics_path = os.path.join(args.out_dir, f"test_metrics_seed{args.seed}.csv")

    for epoch in range(1, args.epochs + 1):
        train_metrics, _, _ = run_epoch(model, train_loader, device, criterion, optimizer=optimizer)
        valid_metrics, _, _ = run_epoch(model, valid_loader, device, criterion, optimizer=None)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "train_f1_flange": train_metrics["f1_flange"],
            "valid_loss": valid_metrics["loss"],
            "valid_accuracy": valid_metrics["accuracy"],
            "valid_macro_f1": valid_metrics["macro_f1"],
            "valid_f1_flange": valid_metrics["f1_flange"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d}: "
            f"Train Loss={train_metrics['loss']:.4f} Acc={100*train_metrics['accuracy']:.2f}% MacroF1={100*train_metrics['macro_f1']:.2f}% | "
            f"Valid Loss={valid_metrics['loss']:.4f} Acc={100*valid_metrics['accuracy']:.2f}% MacroF1={100*valid_metrics['macro_f1']:.2f}%"
        )

        current_metric = valid_metrics["macro_f1"]
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.early_stop_patience:
            print(f"Early stopping na epoch {epoch}. Melhor epoch: {best_epoch}")
            break

    history_df = pd.DataFrame(history)
    history_df.to_csv(history_path, index=False)
    save_curves(history_df, curves_path)

    model.load_state_dict(best_state)
    model.to(device)

    test_metrics, y_true, y_pred = run_epoch(model, test_loader, device, criterion, optimizer=None)
    test_metrics_no_cm = {k: v for k, v in test_metrics.items() if k != "cm"}
    pd.DataFrame([test_metrics_no_cm]).to_csv(test_metrics_path, index=False)
    save_confusion_matrix(test_metrics["cm"], cm_path)

    report = classification_report(
        y_true,
        y_pred,
        target_names=["non_flange", "flange"],
        zero_division=0,
        digits=4,
    )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== TEST METRICS ===\n")
        for k, v in test_metrics_no_cm.items():
            f.write(f"{k}: {v}\n")
        f.write("\n=== CLASSIFICATION REPORT ===\n")
        f.write(report)
        f.write("\n\n=== CONFUSION MATRIX ===\n")
        f.write(np.array2string(test_metrics["cm"]))

    print("\n" + "=" * 60)
    print("TREINO FINALIZADO")
    print("=" * 60)
    print(f"Melhor epoch: {best_epoch}")
    print(f"Best valid macro-F1: {best_metric:.4f}")
    print("\nTeste:")
    for k, v in test_metrics_no_cm.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nModelo salvo em: {model_path}")
    print(f"Histórico: {history_path}")
    print(f"Curvas   : {curves_path}")
    print(f"CM teste : {cm_path}")
    print(f"Relatório: {report_path}")
    print(f"Métricas : {test_metrics_path}")


if __name__ == "__main__":
    main()
