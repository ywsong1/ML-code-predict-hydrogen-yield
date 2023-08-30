#调入库
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import random
import pandas as pd


# 读取数据集
data = pd.read_csv('test.CSV')

# 获取特征和标签
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 归一化处理
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 随机划分训练集和验证集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 超参数组合
max_depth = 10
n_estimators_values = [50, 100, 150, 200, 250]
learning_rate = 0.2

# 初始化结果字典，以max_depth作为键，绝对误差列表作为值
results = {n_estimators: [] for n_estimators in n_estimators_values}

# 循环遍历不同的max_depth值
for n_estimators in n_estimators_values:
    # 初始化随机森林回归模型
    model = GradientBoostingRegressor(max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)

    # 在训练集上训练模型
    model.fit(X_train, y_train)

    # 在预测集上进行预测
    y_pred_pred = model.predict(X_pred)
    pred_mae = mean_absolute_error(y_pred, y_pred_pred)

    # 将绝对误差添加到对应的max_depth键下
    results[n_estimators].extend(np.abs(y_pred - y_pred_pred).tolist())

# 将结果字典转换为DataFrame
results_df = pd.DataFrame(results)

# 保存DataFrame为Excel文件
excel_path = 'D:/xgboost_results1.xlsx'
results_df.to_excel(excel_path, index=False)
print("结果已保存到", excel_path)