import numpy as np
import pandas as pd
import random
import math
import os
from tqdm import tqdm
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import timm
from sklearn.linear_model import LinearRegression # 【新增】线性回归库

# =========================================================================================
# 1. CONFIGURATION
# =========================================================================================
class CONFIG:
    is_debug = False
    seed = 308
    n_folds = 5

    test_csv = "CSIRO/CSIRO_my_5fold_train_csv.csv"
    test_img_path = "CSIRO/train"
    n_workers = os.cpu_count() // 2

    test_batch_size = 1

    # 请确保路径正确
    model_path = "CSIRO/output/2026-01-15_22:56:47_vit_large_patch16_dinov3.lvd1689m_output"
    model_name = "vit_large_patch16_dinov3.lvd1689m"
    if "dinov2" in model_name:
        img_size = [518, 1036]
    elif "eva02" in model_name:
        img_size = [448, 896]
    else:
        img_size = [512, 1024]

    head_out = 5
    DataParallel = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def set_seed(seed=308):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
set_seed(CONFIG.seed)

# =========================================================================================
# 2. DATA LOADING & UTILS
# =========================================================================================
test_all = pd.read_csv(CONFIG.test_csv)
id_and_fold = {}
for i in range(len(test_all)):
    row = test_all.iloc[i, :]
    _id = row.sample_id.split("_")[0]
    _fold = row.fold.item()
    if _id not in id_and_fold.keys():
        id_and_fold[_id] = _fold

test = pd.DataFrame(list(id_and_fold.items()), columns=['sample_id', 'fold'])

def Calculate_Weighted_R2(y_true, y_pred):
    weights = np.array([0.1, 0.1, 0.1, 0.5, 0.2])
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    weighted_sum = np.sum(y_true * weights) 
    total_weight = np.sum(weights) * y_true.shape[0]
    y_bar_w = weighted_sum / total_weight
    ss_res = np.sum(weights * (y_true - y_pred)**2)
    ss_tot = np.sum(weights * (y_true - y_bar_w)**2)
    if ss_tot == 0: return 0.0
    r2 = 1 - (ss_res / ss_tot)
    return r2

# Config Init
cfg = timm.get_pretrained_cfg(CONFIG.model_name)
cfg_dict = cfg.to_dict()
data_config = timm.data.resolve_data_config(pretrained_cfg=cfg_dict)
_mean = data_config['mean']
_std = data_config['std']

print(f"Model: {CONFIG.model_name} | Mean: {_mean} | Std: {_std}")

def transform(img):
    composition = A.Compose([
        A.Resize(CONFIG.img_size[0], CONFIG.img_size[1]),
        A.Normalize(mean=_mean, std=_std),
        ToTensorV2(),
    ])
    return composition(image=img)["image"]

class CSIRODataset(Dataset):
    def __init__(self, df, original_train=test_all, transform=transform):
        super().__init__()
        self.df = df
        self.original_train = original_train
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx, :]
        img_id = row.sample_id
        img_path = os.path.join(CONFIG.test_img_path, img_id + ".jpg")

        img = Image.open(img_path)
        img = np.array(img)
        if self.transform != None:
            img = self.transform(img)

        target_id = ["__Dry_Clover_g", "__Dry_Dead_g", "__Dry_Green_g", "__Dry_Total_g", "__GDM_g"]
        label = []
        for _id in target_id:
            tmp_row = self.original_train[self.original_train["sample_id"] == f"{img_id}{_id}"]["target"].item()
            label.append(tmp_row)
        label = torch.tensor(label, dtype=torch.float32)

        return img, label

def prepare_loaders(df, fold=0):
    df_test = df[df["fold"] == fold]
    test_datasets = CSIRODataset(df=df_test, transform=transform)
    test_loader = DataLoader(test_datasets, batch_size=CONFIG.test_batch_size, num_workers=CONFIG.n_workers, shuffle=False, pin_memory=True)
    return test_loader

# =========================================================================================
# 3. MODEL ARCHITECTURE
# =========================================================================================
class LocalMambaBlock(nn.Module):
    """
    Lightweight Mamba-style block (Gated CNN) from the reference notebook.
    Efficiently mixes tokens with linear complexity.
    """
    def __init__(self, dim, kernel_size=5, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # Depthwise conv mixes spatial information locally
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.gate = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (Batch, Tokens, Dim)
        shortcut = x
        x = self.norm(x)
        # Gating mechanism
        g = torch.sigmoid(self.gate(x))
        x = x * g
        # Spatial mixing via 1D Conv (requires transpose)
        x = x.transpose(1, 2)  # -> (B, D, N)
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # -> (B, N, D)
        # Projection
        x = self.proj(x)
        x = self.drop(x)
        return shortcut + x

class CSIROHead(nn.Module):
    def __init__(self, in_features):
        super(CSIROHead, self).__init__()
        self.fusion = nn.Sequential(
            LocalMambaBlock(in_features, kernel_size=5, dropout=0.1),
            LocalMambaBlock(in_features, kernel_size=5, dropout=0.1)
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_head = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.LayerNorm(in_features // 2),
            nn.LeakyReLU(),
            nn.Dropout(0.2), 
            nn.Linear(in_features // 2, CONFIG.head_out),
        )

    def forward(self, x):
        _tmp = self.fusion(x)
        _tmp_pool = self.pool(_tmp.transpose(1, 2)).flatten(1)
        out = self.out_head(_tmp_pool)
        return out

class CSIROModel(nn.Module):
    def __init__(self):
        super(CSIROModel, self).__init__()
        self.backbone = timm.create_model(model_name=CONFIG.model_name, 
                                          pretrained=False,
                                          global_pool='')

        if "tiny" in CONFIG.model_name:
            in_features = 384
        elif "small" in CONFIG.model_name:
            in_features = 384
        elif "base" in CONFIG.model_name:
            in_features = 768
        elif "large" in CONFIG.model_name:
            in_features = 1024
        elif "huge" in CONFIG.model_name:
            in_features = 1280
        self.head = CSIROHead(in_features)
        self.head.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.01)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        mid = CONFIG.img_size[0]
        x_left = x[:, :, :, :mid]
        x_right = x[:, :, :, mid:]
        _tmp1 = self.backbone(x_left)
        _tmp2 = self.backbone(x_right)
        _tmp = torch.cat([_tmp1, _tmp2], dim=1) # shape: [B, 1536]
        output = self.head(_tmp)
        return output

# =========================================================================================
# 4. LOAD MODELS
# =========================================================================================
all_paths = os.listdir(CONFIG.model_path)
paths = []
for i in range(CONFIG.n_folds):
    _tmp_paths = []
    for _tmp_path in all_paths:
        if _tmp_path.split("_")[0] == str(i+1):
            _tmp_paths.append(_tmp_path)
    best_fold_path = sorted(_tmp_paths, key=lambda x:float(x.split("_")[2]))[-1]
    print(f"Fold {i+1} best path : {best_fold_path}")
    paths.append(best_fold_path)

models = []
if CONFIG.DataParallel:
    device_ids = [0, 1]
    for i in range(CONFIG.n_folds):
        model = CSIROModel()
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        model = model.cuda()
        model.load_state_dict(torch.load(os.path.join(CONFIG.model_path, paths[i])))
        model.eval()
        models.append(model)
else:
    for i in range(CONFIG.n_folds):
        model = CSIROModel()
        model = model.to(CONFIG.device)
        model.load_state_dict(torch.load(os.path.join(CONFIG.model_path, paths[i])))
        model.eval()
        models.append(model)

def Infer(model, test_loader):
    y_preds = []
    y_trues = []
    bar = tqdm(enumerate(test_loader), total=len(test_loader), leave=False)
    with torch.no_grad():
        for step, (images, labels) in bar:
            if CONFIG.DataParallel:
                images = images.cuda().float()
                labels = labels.cuda().float()
            else:
                images = images.to(CONFIG.device, dtype=torch.float)
                labels = labels.to(CONFIG.device, dtype=torch.float)
            output = model(images)
            y_preds.append(output.detach().cpu().numpy())
            y_trues.append(labels.detach().cpu().numpy())
            
    y_preds = np.concatenate(y_preds)
    y_trues = np.concatenate(y_trues)
    return y_preds, y_trues

# =========================================================================================
# 5. INFERENCE LOOP & DATA COLLECTION
# =========================================================================================
y_preds_all = []
y_trues_all = []
fold_indices_all = [] # 【关键】用于存储每个样本属于哪个 Fold

print("\nStarting Inference on 5 Folds...")
for fold in range(0, CONFIG.n_folds):
    print(f"-> Processing Fold {fold+1}...")
    
    test_loader = prepare_loaders(test, fold)
    y_pred, y_true = Infer(models[fold], test_loader)
    
    y_preds_all.append(y_pred)
    y_trues_all.append(y_true)
    
    # 记录当前数据的 Fold 索引，用于后续防泄露验证
    # 创建一个长度等于当前预测数量的数组，填充为当前 fold
    fold_indices_all.append(np.full(len(y_pred), fold))

# 拼接所有数据
oof = np.concatenate(y_preds_all)
true = np.concatenate(y_trues_all)
fold_indices = np.concatenate(fold_indices_all) # shape: [N,] 内容如 [0,0,0... 1,1,1... 4,4,4]

# 计算原始 CV
print("\n" + "="*40)
local_cv = Calculate_Weighted_R2(true, oof)
print(f"Original Local CV  : {local_cv:.5f}")
print("="*40 + "\n")

# =========================================================================================
# 6. LINEAR CALIBRATION (STRICT / NESTED CV)
# =========================================================================================

def apply_calibration_strict(oof_preds, y_true, fold_indices, target_names):
    """
    oof_preds:    [N, 5] (原始 OOF 预测值)
    y_true:       [N, 5] (真实标签)
    fold_indices: [N,]   (样本所属 Fold，用于防泄露)
    """
    print("Applying Leak-Free Linear Calibration (Nested CV)...")
    
    calibrated_oof = np.zeros_like(oof_preds)
    unique_folds = np.unique(fold_indices)
    
    # --- 阶段 1: 严谨验证 (逐折校准) ---
    for val_fold in unique_folds:
        # 定义掩码: 用 "非当前折" 的数据训练，预测 "当前折"
        val_mask = (fold_indices == val_fold)
        train_mask = (fold_indices != val_fold)
        
        for i, target_name in enumerate(target_names):
            # 准备训练数据 (Fold 2,3,4,5)
            X_train = oof_preds[train_mask, i].reshape(-1, 1)
            Y_train = y_true[train_mask, i].reshape(-1, 1)
            
            # 准备验证数据 (Fold 1)
            X_val = oof_preds[val_mask, i].reshape(-1, 1)
            
            # 拟合 & 预测
            lr = LinearRegression(fit_intercept=True)
            lr.fit(X_train, Y_train)
            val_preds_calibrated = lr.predict(X_val).flatten()
            
            # 填入结果 & 截断负数
            calibrated_oof[val_mask, i] = np.maximum(0, val_preds_calibrated)

    # --- 阶段 2: 最终提交参数 (全量训练) ---
    # 这部分参数用于填入你的 submission.py
    print("-" * 60)
    print("FINAL PARAMS FOR TEST SET (Copy these to submission code):")
    print("-" * 60)

    final_scalers = []

    for i, target_name in enumerate(target_names):
        # 使用 全量 OOF 数据
        X_all = oof_preds[:, i].reshape(-1, 1)
        Y_all = y_true[:, i].reshape(-1, 1)
        
        lr_final = LinearRegression(fit_intercept=True)
        lr_final.fit(X_all, Y_all)
        
        scale = lr_final.coef_[0][0]
        bias = lr_final.intercept_[0]
        
        final_scalers.append((scale, bias))
        print(f"Target: {target_name:<15} | Scale: {scale:.5f} | Bias: {bias:.5f}")
    
    print("-" * 60)
    return calibrated_oof, final_scalers

target_names = ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g", "Dry_Total_g", "GDM_g"]

# 执行校准
oof_calibrated, final_params = apply_calibration_strict(oof, true, fold_indices, target_names)

# 计算校准后 CV
calibrated_cv = Calculate_Weighted_R2(true, oof_calibrated)

print("\n" + "="*40)
print(f"Calibrated Local CV: {calibrated_cv:.5f} (Leak-Free)")
print(f"Improvement        : {calibrated_cv - local_cv:+.5f}")
print("="*40)

# 保存结果
np.save("oof_calibrated.npy", oof_calibrated)
np.save("true_labels.npy", true)