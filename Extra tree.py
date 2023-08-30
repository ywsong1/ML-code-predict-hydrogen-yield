import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 导入数据集
data = pd.read_csv("test.CSV")

# 分离自变量和因变量
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 数据集归一化处理
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 随机划分训练集和验证集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义Extra tree回归模型，并使用随机参数进行训练
model = ExtraTreesRegressor(n_estimators=100, max_depth=10, min_samples_split=2, min_samples_leaf=1, learning_rate=0.06)
model.fit(X_train, y_train)

# 使用交叉验证对模型进行测试
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 计算R2、RMSE、MAE、MSE和皮尔逊相关系数等指标
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

train_pearson = np.corrcoef(y_train, y_train_pred)[0, 1]
test_pearson = np.corrcoef(y_test, y_test_pred)[0, 1]

# 将结果存入excel表中
df_train = pd.DataFrame({'Actual': y_train, 'Predicted': y_train_pred})
df_val = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred})

with pd.ExcelWriter(r'D:\宋毅文\graduate student career\小论文\基于机器学习的化学链制氢载氧体的预测\机器学习训练\orginal results\Extra tree_results.xlsx') as writer:
    df_train.to_excel(writer, sheet_name='Training Set')
    df_val.to_excel(writer, sheet_name='Test Set')

# 输出指标结果
print("训练集R2值：", train_r2)
print("验证集R2值：", test_r2)
print("训练集RMSE值：", train_rmse)
print("验证集RMSE值：", test_rmse)
print("训练集MAE值：", train_mae)
print("验证集MAE值：", test_mae)
print("训练集MSE值：", train_mse)
print("验证集MSE值：", test_mse)
print("训练集皮尔逊相关系数：", train_pearson)
print("验证集皮尔逊相关系数：", test_pearson)
