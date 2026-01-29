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



class CONFIG:
    is_debug = False
    seed = 308
    n_folds = 5

    test_csv = "CSIRO/CSIRO_my_5fold_train_csv.csv"
    test_img_path = "CSIRO/train" # (1000, 2000)
    n_workers = os.cpu_count() // 2

    test_batch_size = 16

    model_path = "CSIRO/output/2026-01-13_21:45:33_vit_base_patch16_dinov3.lvd1689m_output"
    model_name = "vit_base_patch16_dinov3.lvd1689m"
    if "dinov2" in model_name:
        img_size = [518, 1036]
    elif "eva02" in model_name:
        img_size = [448, 896]
    else:
        img_size = [512, 1024]
    """
    tf_efficientnet_b0.ns_jft_in1k
    edgenext_base.in21k_ft_in1k
    convnextv2_tiny.fcmae_ft_in22k_in1k
    vit_base_patch14_dinov2.lvd142m
    vit_base_patch16_dinov3.lvd1689m
    """

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
    """
    计算 Kaggle CSIRO Image2Biomass 比赛的加权 R2 分数。
    
    参数:
    y_true: 真实值，形状为 [n_samples, 5]
    y_pred: 预测值，形状为 [n_samples, 5]
    
    列顺序假设:
    0: Dry_Clover_g (w=0.1)
    1: Dry_Dead_g   (w=0.1)
    2: Dry_Green_g  (w=0.1)
    3: Dry_Total_g  (w=0.5)
    4: GDM_g        (w=0.2)
    """
    
    # 1. 定义权重向量
    weights = np.array([0.1, 0.1, 0.1, 0.5, 0.2])
    
    # 2. 确保输入是 numpy 数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 3. 计算全局加权均值 (Global Weighted Mean)
    # 这里的 sum(weights) = 1.0，所以分母实际上就是 样本数 * 1.0
    # 我们利用广播机制将权重应用到每一行
    weighted_sum = np.sum(y_true * weights) 
    total_weight = np.sum(weights) * y_true.shape[0] # weights总和 * 样本数
    y_bar_w = weighted_sum / total_weight
    
    # 4. 计算残差平方和 (SS_res)
    # 公式: sum( w_j * (y_j - y_hat_j)^2 )
    ss_res = np.sum(weights * (y_true - y_pred)**2)
    
    # 5. 计算总离差平方和 (SS_tot)
    # 公式: sum( w_j * (y_j - y_bar_w)^2 )
    # 注意这里减去的是全局加权均值 y_bar_w
    ss_tot = np.sum(weights * (y_true - y_bar_w)**2)
    
    # 6. 计算 R2
    # 避免分母为0的极个别情况
    if ss_tot == 0:
        return 0.0
        
    r2 = 1 - (ss_res / ss_tot)
    
    return r2

# --- 测试用例 ---
# 模拟数据
dummy_true = np.array([
    [5, 16, 36, 54, 42],
    [3, 8, 10, 18, 13]
])
# 假设预测非常接近
dummy_pred = dummy_true + 0.5 

score = Calculate_Weighted_R2(dummy_true, dummy_pred)
print(f"验证集得分: {score:.5f}")

# 使用示例
# 模拟验证集数据
n_valid_samples = 16
# 随机生成验证集真实值和预测值
y_valid_true = np.random.rand(n_valid_samples, 5)
y_valid_pred = np.random.rand(n_valid_samples, 5)
# 计算分数
score = Calculate_Weighted_R2(y_valid_true, y_valid_pred).item()
print(f"score: {score:.5f}")



# 1. 获取配置对象
cfg = timm.get_pretrained_cfg(CONFIG.model_name)

# 2. 【核心修复】先转成字典 (.to_dict()) 再传入
# 这样 resolve_data_config 就能正常使用 .get() 方法了
cfg_dict = cfg.to_dict()
data_config = timm.data.resolve_data_config(pretrained_cfg=cfg_dict)

# 3. 提取结果
_mean = data_config['mean']
_std = data_config['std']

print(f"模型: {CONFIG.model_name}")
print(f"自动获取 Mean: {_mean}")
print(f"自动获取 Std:  {_std}")




# 保持 transform 不变 (只保留 Normalize 和 ToTensor, 这里的 Resize 其实是冗余的但无害)
def transform(img):
    # img 输入是拆分后的 (1000, 1000) numpy array
    composition = A.Compose([
        # Resize 到 512x512
        A.Resize(CONFIG.img_size[0], CONFIG.img_size[0]),
        A.Normalize(mean=_mean, std=_std),
        ToTensorV2(),
    ])
    return composition(image=img)["image"]

class CSIRO_TTA_Dataset(Dataset):
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

        # 1. 读取原始大图 [1000, 2000, 3]
        img = Image.open(img_path)
        img = np.array(img)
        
        # 2. 左右拆分 -> 两张 [1000, 1000, 3]
        mid = img.shape[1] // 2 
        img_left_np = img[:, :mid, :]
        img_right_np = img[:, mid:, :]

        # 3. 分别过 Transform -> 变成 Tensor [3, 512, 512]
        # 注意：这里只做一次 Transform，旋转在 Tensor 上做更高效
        if self.transform is not None:
            tensor_left = self.transform(img_left_np)   # 直接调用
            tensor_right = self.transform(img_right_np) # 直接调用

        images_tta = []
        
        # 4. TTA 循环: 0, 90, 180, 270 度旋转 (在 Tensor 层面)
        # k=0: 0度, k=1: 90度 ...
        for k in range(4):
            # torch.rot90(input, k, dims)
            # dims=[1, 2] 表示在 H 和 W 维度上旋转 (Tensor是 [C, H, W])
            rot_left = torch.rot90(tensor_left, k=k, dims=[1, 2])
            rot_right = torch.rot90(tensor_right, k=k, dims=[1, 2])
            
            # 5. 左右拼接 -> [3, 512, 1024]
            # 在宽度维 (dim=2) 拼接
            img_concat = torch.cat([rot_left, rot_right], dim=2)
            
            images_tta.append(img_concat)
        
        # 6. 堆叠: 返回 [4, 3, 512, 1024]
        images_tensor = torch.stack(images_tta, dim=0)

        # 获取 Label (保持不变)
        target_id = ["__Dry_Clover_g", "__Dry_Dead_g", "__Dry_Green_g", "__Dry_Total_g", "__GDM_g"]
        label = []
        for _id in target_id:
            tmp_row = self.original_train[self.original_train["sample_id"] == f"{img_id}{_id}"]["target"].item()
            label.append(tmp_row)
        label = torch.tensor(label, dtype=torch.float32)

        return images_tensor, label

def prepare_loaders(df, fold=0):
    df_test = df[df["fold"] == fold]
    test_datasets = CSIRO_TTA_Dataset(df=df_test, transform=transform)
    
    # 显存优化：因为每个样本变成了4个，建议 Batch Size 减小
    real_batch_size = max(1, CONFIG.test_batch_size // 4)
    
    test_loader = DataLoader(test_datasets, batch_size=real_batch_size, num_workers=CONFIG.n_workers, shuffle=False, pin_memory=True)
    return test_loader






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
        in_features = 768
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
    
model = CSIROModel()




all_paths = os.listdir(CONFIG.model_path)
all_paths
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
        print(f"Fold_{paths[i]} Load Success!")
else:
    for i in range(CONFIG.n_folds):
        model = CSIROModel()
        model = model.to(CONFIG.device)
        model.load_state_dict(torch.load(os.path.join(CONFIG.model_path, paths[i])))
        model.eval()
        models.append(model)
        print(f"Fold_{paths[i]} Load Success!")




def Infer(model, test_loader):
    y_preds = []
    y_trues = []
    
    bar = tqdm(enumerate(test_loader), total=len(test_loader))
    with torch.no_grad():
        for step, (images, labels) in bar:
            # images: [Batch, 4, 3, 512, 1024]
            batch_size = images.size(0)
            num_tta = images.size(1) # 4
            
            # 1. 融合维度: [B*4, 3, 512, 1024]
            images = images.view(-1, 3, CONFIG.img_size[0], CONFIG.img_size[1])
            
            if CONFIG.DataParallel:
                images = images.cuda().float()
            else:
                images = images.to(CONFIG.device, dtype=torch.float)
                
            # 2. 推理
            output = model(images) # [B*4, 5]
            
            # 3. 还原并取平均
            output = output.view(batch_size, num_tta, -1) # [B, 4, 5]
            output_mean = output.mean(dim=1) # [B, 5]
            
            # 4. 后处理 (Clipping) - 推荐加上
            output_mean = torch.clamp(output_mean, min=0.0)
            
            y_preds.append(output_mean.detach().cpu().numpy())
            y_trues.append(labels.detach().cpu().numpy())
            
    y_preds = np.concatenate(y_preds)
    y_trues = np.concatenate(y_trues)
    return y_preds, y_trues



y_preds = []
y_trues = []
for fold in range(0, CONFIG.n_folds):
    print(f"==================== infer on Fold {fold+1} (Split->Resize->TTA) ====================")
    
    test_loader = prepare_loaders(test, fold)
    y_pred, y_true = Infer(models[fold], test_loader)
    y_preds.append(y_pred)
    y_trues.append(y_true)

oof = np.concatenate(y_preds)
true = np.concatenate(y_trues)

local_cv = Calculate_Weighted_R2(true, oof)
print("Local CV (4x TTA): ", local_cv)