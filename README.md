# Kaggle-CSIRO-Image2Biomass-Prediction-59th-Solution
The solution about CSIRO - Image2Biomass Prediction 59th

**The link to the competition on the Kaggle platform: [CSIRO - Image2Biomass Prediction](https://www.kaggle.com/competitions/csiro-biomass)**

**This directory contains code for training 2-patch models and code for training 8-patch models. It also includes code for local CV inference verification and result analysis.**

**The following section is the same as the solution I published on Kaggle, which you can read here or on Kaggle: [CSIRO 59th Solution](https://www.kaggle.com/competitions/csiro-biomass/writeups/csiro-59th-solution)**

**These are the links to the two notebooks we submitted:**

[CSIRO-infer-base-2x4-GW(Public LB: 0.72 | Private LB: 0.63 | Local CV:0.8610)](https://www.kaggle.com/code/zhiyue666/csiro-infer-base-2x4-gw?scriptVersionId=292394299)

[CSIRO-infer-Ensemble-2p8p(Public LB: 0.71 | Private LB: 0.63 | Local CV:0.8660)](https://www.kaggle.com/code/zhiyue666/csiro-infer-ensemble-2p8p?scriptVersionId=293327244)

## We didn't use overly complicated tricks. Let's start by building a Local CV.
1. We used **StratifiedGroupKFold** + **rare class handling** to construct the CV.
2. We use **Image ID** as the group criterion to ensure that all Target rows corresponding to the same image are strictly divided into the same fold, thus eliminating the data leakage problem of "the same image being in both the training set and the validation set".
3. Regarding **stratification** and **handling of long-tail distribution**: We used **Species** as the stratification basis and implemented a **Rare Class Aggregation** strategy for long-tail distribution. Rare species with fewer than 10 images were uniformly relabeled as **"Other"**. This solved the missing class (NaN) problem in the validation set caused by extreme class imbalance, ensuring that each fold evenly covers common species and rare samples (Hard Examples).
4. This is the CSV file we use after the above processing: [CSIRO-my-train-csv-20251223](https://www.kaggle.com/datasets/zhiyue666/csiro-my-train-csv-20251223)
- Below is a portion of our CV's corresponding PubLB and PriLB, showing an overall positive correlation.

|PublicLB|PrivateLB|Local CV&darr;|
|-----|-----|-----|
|**0.71**|**0.63**|**0.8660**|
|**0.72**|**0.63**|**0.8610**|
|0.70|0.61|0.8511|
|0.69|0.59|0.8436|
|0.65|0.58|0.8423|
|0.68|0.57|0.8338|
|0.61|0.56|0.8189|
|0.58|0.51|0.7758|
|0.54|0.47|0.7312|
|0.51|0.45|0.7153|
|0.28|0.19|0.5809|

## Having discussed the construction of CV, let's talk about the loss function and local evaluation function we used.
- We are using weighted MSELoss, and the following is its code implementation.
```python
class WeightedMSELoss(nn.Module):
    def __init__(self, feature_weights=None):
        super().__init__()
        if feature_weights is None:
            self.register_buffer('feature_weights', 
                               torch.tensor([0.1, 0.1, 0.1, 0.5, 0.2]))
        else:
            self.register_buffer('feature_weights', torch.tensor(feature_weights))
    
    def forward(self, y_pred, y_true):
        weights = self.feature_weights.to(device=y_pred.device, dtype=y_pred.dtype)
        loss = weights * (y_pred - y_true) ** 2
        return loss.sum(dim=1).mean()
```
- The weights above are based on the evaluation method provided by the official competition organizers, so your labels should be sorted in the format of ["__Dry_Clover_g", "__Dry_Dead_g", "__Dry_Green_g", "__Dry_Total_g", "__GDM_g"].
- We also tried using weighted L1Loss, and in our experiments, it performed similarly to weighted MSELoss. Therefore, all subsequent training was based on weighted MSELoss.
- The local evaluation function we used is as follows:
```python
def Calculate_Weighted_R2(y_true, y_pred):
    """
    0: Dry_Clover_g (w=0.1)
    1: Dry_Dead_g   (w=0.1)
    2: Dry_Green_g  (w=0.1)
    3: Dry_Total_g  (w=0.5)
    4: GDM_g        (w=0.2)
    """
    weights = np.array([0.1, 0.1, 0.1, 0.5, 0.2])
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    weighted_sum = np.sum(y_true * weights) 
    total_weight = np.sum(weights) * y_true.shape[0] # weights总和 * 样本数
    y_bar_w = weighted_sum / total_weight

    ss_res = np.sum(weights * (y_true - y_pred)**2)
    ss_tot = np.sum(weights * (y_true - y_bar_w)**2)

    if ss_tot == 0:
        return 0.0
        
    r2 = 1 - (ss_res / ss_tot)
    
    return r2
```

## Next is about model selection and construction.
- In the early stages of our participation in the competition, we used many CNN models (EfficientNet, ConvNext, edgenext) because they didn't need to consider whether the image aspect ratio was 1:1. For the 1:2 images in this competition, they could easily run the code without much consideration.
- After conducting some experiments, we tried using VIT, from Dinov2 to Dinov3. Our experiments showed that VIT consistently outperformed CNNs, perhaps due to VIT's large pre-training capacity. Dinov3 performed better than Dinov2, so in subsequent experiments we used the **"vit_large_patch16_dinov3.lvd1689m"** model. It can be directly called via Timm, which is very convenient.(I love timm🥰)
- For the input to this model, like most people in the code area, we resize the image to (512, 1024), then split the image horizontally into two (512, 512) images, input each image into the model, and then perform feature fusion at the head layer.
- In addition to the **2-patch** approach mentioned above, we also used an **8-patch** approach. The image was resized to (1024, 2048), and then divided into 2×4=8 (512, 512) images, each input into the model for feature fusion.
- For feature fusion at the head layer, we input the [B, L, D] features output from the backbone into LocalMambaBlock for feature fusion, and finally output the predicted 5 regression values ​​directly through two linear layers.
```python
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
            nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Linear(in_features // 2, CONFIG.head_out),
        )

    def forward(self, x):
        _tmp = self.fusion(x)
        _tmp_pool = self.pool(_tmp.transpose(1, 2)).flatten(1)
        out = self.out_head(_tmp_pool)
        return out
```
- The implementation of LocalMambaBlock comes from open-source code in the code section (**thanks to all kagglers for their open-source contributions**).
- The following is the process for submitting the two notebooks.
![process](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F16233502%2Fc1bd56adb3f9fad238f0a78e5e4892f4%2FCSIRO.jpg?generation=1769683093724218&alt=media)
- The post-processing includes Linear Balance and Mass Balance, which you can see in my submission notebook.(Since the results after the first three decimal places are not yet visible, it remains to be seen whether this post-processing is effective.)
- To ensure training stability, we used EMA for training.
```python
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        """Called before training starts: Initialize shadow variables."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Called after each optimizer.step(): Update shadow variables."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                # Compute exponential moving average: 
                # new_avg = (1 - decay) * current_weight + decay * old_avg
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Called before validation: Replace model parameters with shadow parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Called after validation: Restore original parameters to resume training."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
```
- Our training configuration is as follows:
```python
seed = 308
n_folds = 5
epochs = 100
backbone = "vit_large_patch16_dinov3.lvd1689m"
lr_backbone = warmup --> 1e-5 --> 1e-8
lr_head = warmup --> 1e-3 -->  1e-6
use_mixup = True
ema_decay = 0.999

Data Augmentation:
        A.HorizontalFlip(p=0.5), 
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),

Mixup: alpha=0.4, p=0.8
```
### For us, the part that didn't work:
1. **TTA**: We found that using TTA on our local CV did not improve it, but rather decreased it. Therefore, we did not use TTA in our submitted notebooks.
2. **More models fusion**: Similarly, when we used more models to fuse them on local CV, we found that the CV improvement was very limited (after all, all our models were trained on the same dataset, and their prediction distributions were very similar). We found that combining more models did not significantly improve cross-validation performance as much as using a single model for post-processing, so we ultimately chose a single model and a fusion model of 2-patch + 8-patch.
3. **Other data augmentations**: We also used the following data augmentation methods, but they did not yield good results.
```python
A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5),
A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
A.CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.5),
A.OneOf([
    A.RandomBrightnessContrast(p=1.0), 
    A.HueSaturationValue(p=1.0), 
    A.GaussNoise(p=1.0),
], p=0.5),
```



## This is the core of our solution. Thank you for reading.