# ==========================================
# 0. 全局配置与环境设置
# ==========================================

SEQ_LEN = 60         # 过去60天的数据预测下一天
FUTURE_DAYS = 30     # 预测未来多少天
BATCH_SIZE = 64      # 批大小
EPOCHS = 30          # 训练轮数
LR = 0.001           # 学习率
# ------------------------------------

'''
本项目构建了一个基于 Stacking（堆叠泛化） 策略的混合时间序列预测框架。
代码旨在解决传统单一模型在复杂时序数据上泛化能力不足的问题，
通过结合 LSTM（长短期记忆网络）、Transformer 和 TCN（时间卷积网络）三种不同架构的深度学习模型提取时序特征，
并利用 XGBoost 作为元学习器（Meta-Learner）进行结果融合，最终实现对目标变量（Target）的高精度滚动预测。
'''

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import weight_norm
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
import time

start = time.perf_counter()

# 设置随机种子以复现结果
warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)
plt.rcParams['font.family'] = 'sans-serif' 
plt.rcParams['axes.unicode_minus'] = False

# *** 在这里定义 device，确保全局可用 ***
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

FILE_PATH = 'C://Users//Limin Feng//Code_Notebook//2025//China_Regions_Precipitation_Tianwen.xlsx'  
plot_output='C://Users//Limin Feng//Code_Notebook//2025//'
TARGET_COL = 'NC'   # 预测目标变量
DATE_COL = 'time'        # 时间列名

vas_tianwen=['Moon_Phase','Moon_Light','Moon_Earth','Sun_Earth','Venus_Earth',
			 'Mercury_Earth','Mars_Earth','Jupiter_Earth','Saturn_Earth',
			 'Lon_Earth','Lon_Moon','Lon_Sun','Lon_Mercury','Lon_Venus',
			 'Lon_Mars','Lon_Jupiter','Lon_Saturn','Mercury_Mars','Mercury_Venus','Mars_Venus']

vas_ganzhi=['年干配数', '年支天数', '年支地数', '月干配数', '月支天数', '月支地数', 
			'日干配数', '日支天数', '日支地数']

vas_meihua=['本卦编码_0_encoded', '本卦编码_6_encoded', '本卦编码_12_encoded', '本卦编码_18_encoded', 
			'变卦编码_0_encoded', '变卦编码_6_encoded', '变卦编码_12_encoded', '变卦编码_18_encoded']

vas_Pythagoras=['Pythagoras_0', 'Pythagoras_1', 'Pythagoras_2']

vas_area=['NEC','IM','NWC','NC','QT','SWC','SCC']

va_Y=['NC']

vas_X = vas_tianwen+vas_ganzhi+vas_meihua+vas_Pythagoras

FEATURE_COLS = va_Y+vas_X #

# ==========================================
# 1. 高效数据读取与预处理
# ==========================================

print("1. 正在读取数据...")
df = pd.read_excel(FILE_PATH)

df[DATE_COL] = pd.to_datetime(df[DATE_COL])
df = df.sort_values(DATE_COL).reset_index(drop=True)

# 找到有效历史数据的截止点 (即 TARGET_COL 不为空的最后一行)
valid_data_mask = df[TARGET_COL].notna()
last_valid_idx = df[valid_data_mask].index[-1]
print(f"  - 历史数据截止日期: {df.loc[last_valid_idx, DATE_COL].date()}")

# 归一化逻辑：
# 1. 仅用历史有效数据 fit (防止未来信息泄露)
scaler = StandardScaler()
train_subset = df.loc[:last_valid_idx, FEATURE_COLS].values
scaler.fit(train_subset)

# 2. 对全量数据 (含未来的X) 进行 transform
data_scaled = scaler.transform(df[FEATURE_COLS].values)

# 3. 单独准备一个 scaler_y 用于最后反归一化结果
scaler_y = StandardScaler()
scaler_y.fit(df.loc[:last_valid_idx, [TARGET_COL]].values)

# 记录目标列在特征矩阵中的索引，用于后续填充
target_col_idx = FEATURE_COLS.index(TARGET_COL)

# ==========================================
# 2. 数据加载器 (Dataset)
# ==========================================
class TimeSeriesDataset(Dataset):
    def __init__(self, scaled_matrix, seq_len, end_idx):
        # 传入全量矩阵，但只允许访问到 end_idx
        self.data = torch.tensor(scaled_matrix, dtype=torch.float32)
        self.seq_len = seq_len
        self.end_idx = end_idx
        
    def __len__(self):
        return self.end_idx - self.seq_len

    def __getitem__(self, idx):
        # 动态切片：[idx, idx+seq_len]
        x_seq = self.data[idx : idx + self.seq_len]
        y_val = self.data[idx + self.seq_len, target_col_idx]
        
        # 安全性检查：如果有NaN转为0 (理论上训练集不该有NaN)
        if torch.isnan(x_seq).any():
             x_seq = torch.nan_to_num(x_seq)
        if torch.isnan(y_val):
             y_val = torch.tensor(0.0)
             
        return x_seq, y_val.unsqueeze(0)

# 划分训练集 (80%的历史数据)
train_size = int(last_valid_idx * 0.8)
train_dataset = TimeSeriesDataset(data_scaled, SEQ_LEN, train_size)

# 验证集 (剩下的20%历史数据)
# 为了简单，我们这里验证集只取后半段
val_dataset = TimeSeriesDataset(data_scaled, SEQ_LEN, last_valid_idx)
val_indices = list(range(train_size, last_valid_idx - SEQ_LEN))
val_subset = torch.utils.data.Subset(val_dataset, val_indices)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# 3. 模型定义 (含 TCN Chomp1d 修复)
# ==========================================

# --- Chomp1d: 用于裁剪 TCN 多余的 padding ---
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding) # 裁剪
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding) # 裁剪
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_size, num_channels=[64, 64], kernel_size=3, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_ch = input_size if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, stride=1, dilation=dilation_size,
                                        padding=(kernel_size-1) * dilation_size, dropout=dropout))
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        # x: (Batch, Seq, Feat) -> (Batch, Feat, Seq)
        x = x.permute(0, 2, 1) 
        y = self.network(x)
        return self.fc(y[:, :, -1])

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        return self.fc(x[:, -1, :])

# ==========================================
# 4. 训练函数
# ==========================================
def train_engine(model, loader, epochs=10, name="Model"):
    # 将模型移动到全局定义的 device
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    print(f"--- 开始训练 {name} ---")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch in loader:
            # 将数据移动到 device
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch+1) % 5 == 0:
            print(f"  Epoch {epoch+1}: Loss = {total_loss/len(loader):.6f}")
    return model

# 实例化模型
input_dim = len(FEATURE_COLS)

lstm = LSTMModel(input_dim)
lstm = train_engine(lstm, train_loader, EPOCHS, "LSTM")

transformer = TransformerModel(input_dim)
transformer = train_engine(transformer, train_loader, EPOCHS, "Transformer")

tcn = TCN(input_dim)
tcn = train_engine(tcn, train_loader, EPOCHS, "TCN") # 此时TCN已包含Chomp1d，不会报错

# ==========================================
# 5. Stacking 融合
# ==========================================
print("5. 训练 Stacking 融合模型 (XGBoost)...")

lstm.eval(); transformer.eval(); tcn.eval()
val_preds = []
val_targets = []

with torch.no_grad():
    for x_val, y_val in val_loader:
        x_val = x_val.to(device)
        p1 = lstm(x_val).cpu().numpy()
        p2 = transformer(x_val).cpu().numpy()
        p3 = tcn(x_val).cpu().numpy()
        
        val_preds.append(np.hstack([p1, p2, p3]))
        val_targets.append(y_val.numpy())

val_preds = np.concatenate(val_preds, axis=0)
val_targets = np.concatenate(val_targets, axis=0)

xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05)
xgb_model.fit(val_preds, val_targets.ravel())

# ==========================================
# 6. 未来滚动预测 (关键：利用已有未来X)
# ==========================================
print(f"6. 开始滚动预测未来 {FUTURE_DAYS} 天...")

# 复制全量数据 (含历史Y和未来X) 用于模拟
sim_data = data_scaled.copy() 
future_preds_inv = []

# 起始点：历史数据的最后一天
start_idx = last_valid_idx

for i in range(FUTURE_DAYS):
    # 当前预测的时间点索引: last_valid_idx + 1 + i
    curr_idx = start_idx + 1 + i
    
    # 提取窗口 [t-SEQ_LEN, t]
    # 这个窗口包含了：
    # 1. 部分历史真实数据
    # 2. 部分之前循环预测填入的 Y
    # 3. Excel 中自带的真实 X (Open, Macro等)
    window = sim_data[curr_idx - SEQ_LEN : curr_idx]
    
    # 转换为Tensor并预测
    window_t = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
    
    p1 = lstm(window_t).detach().cpu().numpy()
    p2 = transformer(window_t).detach().cpu().numpy()
    p3 = tcn(window_t).detach().cpu().numpy()
    
    # 融合
    stack_in = np.hstack([p1, p2, p3])
    pred_scaled = xgb_model.predict(stack_in)[0]
    
    # 核心：将预测的 Y 填入 sim_data 的目标列
    sim_data[curr_idx, target_col_idx] = pred_scaled
    
    # 保存反归一化结果
    pred_real = scaler_y.inverse_transform([[pred_scaled]])[0,0]
    future_preds_inv.append(pred_real)

# ==========================================
# 7. 绘图
# ==========================================
print("7. 正在绘图...")

# 历史数据 (取最近100天展示)
disp_len = 100
hist_dates = df.loc[last_valid_idx-disp_len : last_valid_idx, DATE_COL]
hist_vals = df.loc[last_valid_idx-disp_len : last_valid_idx, TARGET_COL].values

# 未来日期
fut_dates = df.loc[last_valid_idx+1 : last_valid_idx+FUTURE_DAYS, DATE_COL]
if len(fut_dates) < FUTURE_DAYS:
    # 补全日期以防万一
    last_d = pd.to_datetime(hist_dates.iloc[-1])
    fut_dates = pd.date_range(last_d + pd.Timedelta(days=1), periods=FUTURE_DAYS)

# 拼接数据以绘制连续线条
join_date = pd.concat([hist_dates.iloc[[-1]], fut_dates])
join_val = np.concatenate([[hist_vals[-1]], future_preds_inv])

plt.figure(figsize=(14, 6))
plt.plot(hist_dates, hist_vals, color='#FF4500', label='Historical Observation(Real)', linewidth=2)
plt.plot(join_date, join_val, color='#1E90FF', label='Future Prediction (Using Future X)', 
         linewidth=2.5, linestyle='--')

plt.axvline(x=hist_dates.iloc[-1], color='gray', linestyle=':', alpha=0.8)
plt.text(hist_dates.iloc[-1], hist_vals[-1], ' Today', fontsize=10, verticalalignment='bottom')

plt.title(f'Prediction of {TARGET_COL} using Hybrid Models', fontsize=16)
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(os.path.join(plot_output,'Prediction & Future Simulation.png'), dpi=300) 

print("绘图完成。蓝色部分延伸到了未来，红色部分为已发生的真实走势。")

# ==========================================
# 8. 绘制残差 Stacking 融合预测图
# ==========================================
print("正在绘制残差 Stacking 融合验证图...")

# 引入 seaborn 以匹配附件图表的风格 (如果没有安装，会自动回退到 matplotlib 默认风格)
try:
    import seaborn as sns
    sns.set_theme(style="darkgrid")
except ImportError:
    pass

# 1. 使用训练好的 XGBoost 模型重新对验证集进行预测
# val_preds: 验证集上 LSTM/Transformer/TCN 的输出堆叠 (Input)
# val_targets: 验证集的真实标签 (Target)
fusion_preds_scaled = xgb_model.predict(val_preds)

# 2. 反归一化数据 (将 Scaled 数据还原为真实股价/数值)
val_true_inv = scaler_y.inverse_transform(val_targets.reshape(-1, 1)).flatten()
val_pred_inv = scaler_y.inverse_transform(fusion_preds_scaled.reshape(-1, 1)).flatten()

# 3. 提取验证集对应的时间轴
# 逻辑：验证集 indices 是 range(train_size, ...)，Dataset 取值是 idx + SEQ_LEN
val_date_indices = [i + SEQ_LEN for i in val_indices]
val_dates = df.loc[val_date_indices, DATE_COL]

# 4. 绘图 (仿照附件风格：紫色预测，红橙色真实，灰色背景网格)
plt.figure(figsize=(12, 6), dpi=300)

# 设置中文字体 (尝试常见中文字体，防止乱码)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False 

# 绘制线条
plt.plot(val_dates, val_true_inv, label='真实值', color='#FF6347', linewidth=1.5) # 番茄红/橙色
plt.plot(val_dates, val_pred_inv, label='融合预测', color='#8A2BE2', linewidth=1.5, alpha=0.9) # 蓝紫色

# 标签与标题
plt.title("残差 stacking 融合预测", fontsize=14)
plt.xlabel("时间", fontsize=12)
plt.ylabel("数值", fontsize=12)
plt.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.8)
plt.grid(True, which='major', linestyle='-', alpha=0.6)

# 保存
save_name = os.path.join(plot_output, 'Residual_Stacking_Fusion_Validation.png')
plt.tight_layout()
plt.savefig(save_name, dpi=300)
print(f"残差融合预测图已保存至: {save_name}")

end = time.perf_counter()
print(f"耗时: {end - start:.1f} 秒")

'''
 XGBoost 模型 的输入并不是原始特征（如开盘价、成交量），而是基模型（LSTM, Transformer, TCN）的预测结果。
'''
# ==========================================
# 9. 补充绘图：训练集拟合对比 & 模型重要性
# ==========================================
print("\n9. 正在补充绘制：训练集拟合详情与模型重要性分析...")

# ------------------------------------------------------------------
# 9.1 训练集拟合效果对比 (Training Set Fit)
# ------------------------------------------------------------------
# 注意：原 train_loader 是 shuffle=True 的，绘图需要按时间顺序，因此重建一个无序 loader
train_loader_seq = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

train_preds_list = []
train_targets_list = []

lstm.eval(); transformer.eval(); tcn.eval()

with torch.no_grad():
    for x_train, y_train in train_loader_seq:
        x_train = x_train.to(device)
        
        # 1. 基模型预测
        p1 = lstm(x_train).cpu().numpy()
        p2 = transformer(x_train).cpu().numpy()
        p3 = tcn(x_train).cpu().numpy()
        
        # 2. 堆叠 (Stacking)
        train_preds_list.append(np.hstack([p1, p2, p3]))
        train_targets_list.append(y_train.numpy())

# 拼接所有批次
train_stack_input = np.concatenate(train_preds_list, axis=0)
train_targets_raw = np.concatenate(train_targets_list, axis=0)

# 3. 通过 XGBoost 得到最终训练集拟合值
train_final_pred_scaled = xgb_model.predict(train_stack_input)

# 4. 反归一化
train_pred_inv = scaler_y.inverse_transform(train_final_pred_scaled.reshape(-1, 1)).flatten()
train_true_inv = scaler_y.inverse_transform(train_targets_raw.reshape(-1, 1)).flatten()

# 5. 生成对应的时间轴
# 训练集的数据范围是从 0 到 train_size，但由于 seq_len 的存在，
# 实际产出的预测是从第 seq_len 天开始的
train_date_indices = range(SEQ_LEN, train_size)
# 确保索引长度匹配 (防止整除截断导致的微小误差)
limit = min(len(train_date_indices), len(train_true_inv))
train_dates = df.loc[list(train_date_indices)[:limit], DATE_COL]

# 绘图：训练集拟合对比
plt.figure(figsize=(14, 6), dpi=300)
plt.plot(train_dates, train_true_inv[:limit], label='Precipitation (Observed)', color='#2E8B57', alpha=0.7)
plt.plot(train_dates, train_pred_inv[:limit], label='Precipitation (Fitted)', color='#FF8C00', alpha=0.7, linestyle='--')
plt.title(f"Training Set Fit", fontsize=16)
plt.xlabel("Date")
plt.ylabel(f"{TARGET_COL} Precipitation (mm)")
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plot_output, 'Training_Set_Fit_Analysis.png'), dpi=300)
print("  - 训练集拟合对比图已保存。")

# ------------------------------------------------------------------
# 9.2 Stacking 元模型特征重要性 (Model Importance)
# ------------------------------------------------------------------
# 说明：因为 XGBoost 的输入是三个基模型的预测值，所以这里的“特征重要性”
# 代表的是：LSTM、Transformer、TCN 三个模型在最终决策中的权重/贡献度。

models_names = ['LSTM', 'Transformer', 'TCN']
importances = xgb_model.feature_importances_
indices = np.argsort(importances)[::-1] # 降序排列

plt.figure(figsize=(8, 6), dpi=300)
# 使用 Seaborn 调色板或自定义颜色
colors = ['#4c72b0', '#55a868', '#c44e52']

plt.bar(range(len(importances)), importances[indices], align='center', color=[colors[i] for i in indices])
plt.xticks(range(len(importances)), [models_names[i] for i in indices], fontsize=12)
plt.title("Stacking 融合模型权重分析 (Which Model Contributes More?)", fontsize=15)
plt.ylabel("Importance Score (Gain)", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# 在柱状图上方显示数值
for i, v in enumerate(importances[indices]):
    plt.text(i, v + 0.01, f"{v:.4f}", ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(plot_output, 'Model_Importance_Analysis.png'), dpi=300)
print("  - 模型重要性分析图已保存。")

# ==========================================
# 9.1 (修订版) 测试集(验证集)拟合效果对比与指标计算
# ==========================================
print("\n9.1 正在绘制：测试集（验证集）拟合详情...")

# 1. 确保模型处于评估模式
lstm.eval(); transformer.eval(); tcn.eval()

test_preds_list = []
test_targets_list = []

# 2. 遍历验证集 (val_loader 已经是 shuffle=False 的)
with torch.no_grad():
    for x_test, y_test in val_loader:
        x_test = x_test.to(device)
        
        # 基模型预测
        p1 = lstm(x_test).cpu().numpy()
        p2 = transformer(x_test).cpu().numpy()
        p3 = tcn(x_test).cpu().numpy()
        
        # 堆叠输入
        test_preds_list.append(np.hstack([p1, p2, p3]))
        test_targets_list.append(y_test.numpy())

# 3. 拼接数据
test_stack_input = np.concatenate(test_preds_list, axis=0)
test_targets_raw = np.concatenate(test_targets_list, axis=0)

# 4. Stacking (XGBoost) 最终预测
# 注意：这里调用的是 predict，不是 fit
test_final_pred_scaled = xgb_model.predict(test_stack_input)

# 5. 反归一化 (还原为真实数值)
test_pred_inv = scaler_y.inverse_transform(test_final_pred_scaled.reshape(-1, 1)).flatten()
test_true_inv = scaler_y.inverse_transform(test_targets_raw.reshape(-1, 1)).flatten()

# 6. 计算评估指标 (RMSE, MAE, R2)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(test_true_inv, test_pred_inv)
rmse = np.sqrt(mse)
mae = mean_absolute_error(test_true_inv, test_pred_inv)
r2 = r2_score(test_true_inv, test_pred_inv)

print(f"--- 测试集评估指标 ---")
print(f"  RMSE (均方根误差): {rmse:.4f}")
print(f"  MAE  (平均绝对误差): {mae:.4f}")
print(f"  R2   (拟合优度):     {r2:.4f}")

# 7. 生成对应的时间轴
# 逻辑：val_indices 是 range(train_size, last_valid_idx - SEQ_LEN)
# Dataset 取出的 y 是在 idx + SEQ_LEN 处
test_date_indices = [i + SEQ_LEN for i in val_indices]

# 截断以匹配实际预测长度 (防止DataLoader drop_last等造成的细微长度差异)
limit = min(len(test_date_indices), len(test_true_inv))
test_dates = df.loc[test_date_indices[:limit], DATE_COL]

# 8. 绘图
plt.figure(figsize=(14, 6), dpi=300)

# 绘制真实值
plt.plot(test_dates, test_true_inv[:limit], label='Precipitation (Actual)', color='skyblue',linewidth=2, alpha=0.8)#, color='#2E8B57'
# 绘制预测值
plt.plot(test_dates, test_pred_inv[:limit], label='Precipitation (Predicted)', color='#DC143C', linewidth=1.5, linestyle='--', alpha=0.9)

plt.title(f"Test Set Evaluation ($R^2$={r2:.2f}, RMSE={rmse:.2f})", fontsize=16)
plt.xlabel("Date")
plt.ylabel(f"{TARGET_COL} Precipitation (mm)")
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()

save_path = os.path.join(plot_output, 'Test_Set_Prediction_Comparison.png')
plt.savefig(save_path, dpi=300)
print(f"  - 测试集拟合对比图已保存至: {save_path}")

# ==========================================
# 补充：训练集表现定量评估 (Metrics on Training Set)
# ==========================================
print("\n9.2 正在计算训练集评估指标...")

# 1. 创建有序的训练集加载器 (Shuffle=False)
# 必须使用不打乱的顺序，以保证特征和标签一一对应，方便分析
train_loader_eval = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

train_preds_list = []
train_targets_list = []

# 确保模型在评估模式
lstm.eval(); transformer.eval(); tcn.eval()

with torch.no_grad():
    for x_train, y_train in train_loader_eval:
        x_train = x_train.to(device)
        
        # 1. 基模型提取特征 (Base Models Inference)
        p1 = lstm(x_train).cpu().numpy()
        p2 = transformer(x_train).cpu().numpy()
        p3 = tcn(x_train).cpu().numpy()
        
        # 2. 堆叠 (Stacking)
        train_preds_list.append(np.hstack([p1, p2, p3]))
        train_targets_list.append(y_train.numpy())

# 拼接数据
train_stack_input = np.concatenate(train_preds_list, axis=0)
train_targets_raw = np.concatenate(train_targets_list, axis=0)

# 3. 通过 Stacking 元模型 (XGBoost) 获取最终预测
# 注意：这里我们看 XGBoost 能否很好地处理它没见过的“训练集基模型输出”
# 虽然 XGBoost 是在验证集输出上训练的，但它应该能泛化到训练集上
train_final_pred_scaled = xgb_model.predict(train_stack_input)

# 4. 反归一化 (Inverse Transform)
train_pred_inv = scaler_y.inverse_transform(train_final_pred_scaled.reshape(-1, 1)).flatten()
train_true_inv = scaler_y.inverse_transform(train_targets_raw.reshape(-1, 1)).flatten()

# 5. 计算评估指标
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

train_mse = mean_squared_error(train_true_inv, train_pred_inv)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(train_true_inv, train_pred_inv)
train_r2 = r2_score(train_true_inv, train_pred_inv)

print(f"--- 训练集评估指标 (Training Set Metrics) ---")
print(f"  RMSE (均方根误差): {train_rmse:.4f}")
print(f"  MAE  (平均绝对误差): {train_mae:.4f}")
print(f"  R2   (拟合优度):     {train_r2:.4f}")

# 6. 简单的过拟合/欠拟合诊断逻辑
print("-" * 30)
print("【模型状态诊断】")
if train_r2 > 0.90 and r2 < 0.5: 
    # 注意：这里的 r2 变量引用的是上一段代码中测试集的 r2，请确保先运行了测试集评估
    print("  警告：可能存在严重过拟合 (Overfitting)。\n  建议：增加 Dropout，减小模型层数，或增大正则化系数。")
elif train_r2 < 0.5 and r2 < 0.5:
    print("  警告：可能存在欠拟合 (Underfitting)。\n  建议：增加模型复杂度，增加特征数量，或增加训练 Epoch。")
else:
    print("  模型状态相对正常。请对比训练集与测试集的 RMSE 差异，越小越好。")
print("-" * 30)

print("\n所有补充绘图任务完成。")
