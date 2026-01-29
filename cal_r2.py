import numpy as np
from sklearn.linear_model import LinearRegression # [New Import]

oof_path = "CSIRO/output/2026-01-25_15:57:46_vit_large_patch16_dinov3.lvd1689m_output/oof.npy"

true_path = "CSIRO/true.npy"

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

def apply_mass_balance_numpy(oof_array):
    """
    输入: oof_array (N, 5)
    列顺序假设: 0:Clover, 1:Dead, 2:Green, 3:Total, 4:GDM
    """
    vals = oof_array.T  # 转置为 (5, N) 以便矩阵乘法
    
    # --- 定义约束矩阵 C (Cx = 0) ---
    # 变量顺序 x = [Clover, Dead, Green, Total, GDM]
    # 约束 1: Green + Clover - GDM = 0  =>  1*C + 0*D + 1*G + 0*T - 1*M = 0
    # 约束 2: GDM + Dead - Total = 0    =>  0*C + 1*D + 0*G - 1*T + 1*M = 0
    
    C = np.array([
        [1, 0, 1,  0, -1], # row 1: G + C = GDM
        [0, 1, 0, -1,  1]  # row 2: GDM + D = Total
    ])
    
    # --- 计算投影矩阵 P = I - C_T * (C * C_T)^-1 * C ---
    C_T = C.T
    # 计算 (C * C_T) 的逆
    inv_CCt = np.linalg.inv(C @ C_T)
    # 计算 P
    P = np.eye(5) - C_T @ inv_CCt @ C
    
    # --- 应用投影 ---
    # vals_corrected = P * vals
    vals_corrected = P @ vals
    
    # 转置回来 (N, 5)
    vals_corrected = vals_corrected.T
    
    # --- 后处理 ---
    # 投影可能会产生微小的负数，截断为 0
    vals_corrected = np.maximum(0, vals_corrected)
    
    return vals_corrected

def apply_calibration(oof_preds, y_true, target_names):
    """
    oof_preds: [N, 5] (模型预测值)
    y_true:    [N, 5] (真实值)
    """
    print("Applying Linear Calibration...")
    
    calibrated_oof = np.zeros_like(oof_preds)
    calibration_models = {} # 如果需要保存模型供测试集使用，可以存这里
    
    # 对 5 个目标分别训练线性回归
    for i, target_name in enumerate(target_names):
        # 准备数据: X=Pred, Y=True
        X = oof_preds[:, i].reshape(-1, 1)
        Y = y_true[:, i].reshape(-1, 1)
        
        # 训练线性回归 (y = ax + b)
        # fit_intercept=True 允许学习偏差 b (Shift)
        lr = LinearRegression(fit_intercept=True)
        lr.fit(X, Y)
        
        # 获取系数
        scale = lr.coef_[0][0]  # 斜率 a
        bias = lr.intercept_[0] # 截距 b
        
        # 应用校准
        # New_Pred = a * Old_Pred + b
        calibrated_col = lr.predict(X).flatten()
        
        # 物理约束: 截断负数
        calibrated_col = np.maximum(0, calibrated_col)
        
        calibrated_oof[:, i] = calibrated_col
        
        print(f"  Target: {target_name:<15} | Scale(a): {scale:.4f} | Bias(b): {bias:.4f}")
        
    return calibrated_oof

if __name__ == "__main__":
    # ["__Dry_Clover_g", "__Dry_Dead_g", "__Dry_Green_g", "__Dry_Total_g", "__GDM_g"]
    oof = np.load(oof_path)
    oof[oof < 0] = 0.0

    true = np.load(true_path)

    local_cv = Calculate_Weighted_R2(true, oof)
    print("Local CV : ", local_cv)

    # 3. 应用物理约束修正
    print("Applying Mass Balance Constraint...")
    oof_corrected = apply_mass_balance_numpy(oof)
    
    # 4. 计算修正后分数
    corrected_cv = Calculate_Weighted_R2(true, oof_corrected)
    print(f"[Corrected] CV Score: {corrected_cv:.6f}")

    


    # 定义目标名称
    target_names = ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g", "Dry_Total_g", "GDM_g"]

    # 执行校准
    oof_calibrated = apply_calibration(oof, true, target_names)

    # 校准后分数
    calibrated_cv = Calculate_Weighted_R2(true, oof_calibrated)
    print("\n" + "="*40)
    print(f"Calibrated Local CV : {calibrated_cv:.5f}")
    diff = calibrated_cv - local_cv
    print(f"Improvement         : {diff:+.5f}")
    print("="*40)

    # 3. 应用物理约束修正
    print("Applying Mass Balance Constraint...")
    oof_corrected = apply_mass_balance_numpy(oof_calibrated)
    
    # 4. 计算修正后分数
    corrected_cv = Calculate_Weighted_R2(true, oof_corrected)
    print(f"[Corrected] CV Score: {corrected_cv:.6f}")


