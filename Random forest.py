from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd

# 读取数据集
data = np.loadtxt('test.CSV', delimiter=',', skiprows=1)
X = data[:, :-1]
y = data[:, -1]

# 数据归一化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 随机划分训练集和验证集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 设置随机超参数
n_estimators = 100
max_depth = np.random.randint(2, 10)
max_features = np.random.choice(['sqrt', 'log2'])

# 构建Random Forest回归模型
rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, random_state=42)

# 交叉验证评估模型性能
scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-scores)
print("Cross-validation RMSE scores:", rmse_scores)
print("Mean RMSE score:", rmse_scores.mean())

# 训练模型
rf.fit(X_train, y_train)

# 预测并评估模型性能
y_train_pred = rf.predict(X_train)
y_test_pred = rf.predict(X_test)
print("R2 score on training set:", r2_score(y_train, y_train_pred))
print("R2 score on test set:", r2_score(y_test, y_test_pred))
print("RMSE on training set:", np.sqrt(mean_squared_error(y_train, y_train_pred)))
print("RMSE on test set:", np.sqrt(mean_squared_error(y_test, y_test_pred)))
print("MAE on training set:", mean_absolute_error(y_train, y_train_pred))
print("MAE on test set:", mean_absolute_error(y_test, y_test_pred))
print("MSE on training set:", mean_squared_error(y_train, y_train_pred))
print("MSE on test set:", mean_squared_error(y_test, y_test_pred))
print("Pearson correlation coefficient on test set:", np.corrcoef(y_test, y_test_pred)[0, 1])

# 将结果存入excel表中
df_train = pd.DataFrame({'Actual': y_train, 'Predicted': y_train_pred})
df_val = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred})

with pd.ExcelWriter(r'D:\宋毅文\graduate student career\小论文\基于机器学习的化学链制氢载氧体的预测\机器学习训练\orginal results\Random forest_results.xlsx') as writer:
    df_train.to_excel(writer, sheet_name='Training Set')
    df_val.to_excel(writer, sheet_name='Test Set')