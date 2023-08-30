import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 读取数据集
data = pd.read_csv('test.CSV')

# 获取特征和标签
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 对特征进行归一化处理
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 随机划分训练集和验证集
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# 设置超参数，随机寻找一组参数
n_estimators = np.random.randint(50, 200)
max_depth = np.random.randint(2, 10)
min_samples_split = np.random.randint(2, 20)
learning_rate = np.random.uniform(0.01, 0.1)

# 创建GBR回归模型
model = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, learning_rate=learning_rate, random_state=42)

# 训练模型
model.fit(X_train, y_train)

# 交叉验证评估模型
scores = cross_val_score(model, X_train, y_train, cv=5)

# 在训练集和验证集上评估模型
y_train_pred = model.predict(X_train)
y_valid_pred = model.predict(X_valid)

r2_train = r2_score(y_train, y_train_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
mae_train = mean_absolute_error(y_train, y_train_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
corr_train = np.corrcoef(y_train, y_train_pred)[0][1]

r2_valid = r2_score(y_valid, y_valid_pred)
rmse_valid = np.sqrt(mean_squared_error(y_valid, y_valid_pred))
mae_valid = mean_absolute_error(y_valid, y_valid_pred)
mse_valid = mean_squared_error(y_valid, y_valid_pred)
corr_valid = np.corrcoef(y_valid, y_valid_pred)[0][1]

# 将结果存入excel表中
df_train = pd.DataFrame({'Actual': y_train, 'Predicted': y_train_pred})
df_val = pd.DataFrame({'Actual': y_valid, 'Predicted': y_valid_pred})

with pd.ExcelWriter(r'D:\宋毅文\graduate student career\小论文\基于机器学习的化学链制氢载氧体的预测\机器学习训练\orginal results\GBR_results.xlsx') as writer:
    df_train.to_excel(writer, sheet_name='Training Set')
    df_val.to_excel(writer, sheet_name='Test Set')

print('Random parameters:')
print(f'n_estimators: {n_estimators}')
print(f'max_depth: {max_depth}')
print(f'min_samples_split: {min_samples_split}')
print(f'learning_rate: {learning_rate}')

print('\nCross-validation scores:', scores)
print(f'Mean cross-validation score: {np.mean(scores)}')

print('\nTraining set:')
print(f'R2 score: {r2_train:.3f}')
print(f'RMSE: {rmse_train:.3f}')
print(f'MAE: {mae_train:.3f}')
print(f'MSE: {mse_train:.3f}')
print(f'Pearson correlation coefficient: {corr_train:.3f}')

print('\nValidation set:')
print(f'R2 score: {r2_valid:.3f}')
print(f'RMSE: {rmse_valid:.3f}')
print(f'MAE: {mae_valid:.3f}')
print(f'MSE: {mse_valid:.3f}')
print(f'Pearson correlation coefficient: {corr_valid:.3f}')
