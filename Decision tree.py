import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

# 读取数据集
data = pd.read_csv('test.CSV')

# 获取特征和标签
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 将数据集随机划分为训练集和验证集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 对数据集进行归一化处理
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 随机设置超参数
max_depth = np.random.randint(2, 10)
min_samples_split = np.random.randint(2, 10)
min_samples_leaf = np.random.randint(1, 5)

# 构建Decision Tree回归模型
model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)

# 训练模型
model.fit(X_train, y_train)

# 使用交叉验证的方法对模型进行评估
cv_scores = cross_val_score(model, X_train, y_train, cv=5)

# 在训练集和验证集上进行预测，并计算R2、RMSE、MAE、MSE和皮尔逊相关系数等参数
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
pearson_train = pearsonr(y_train, y_train_pred)[0]
pearson_test = pearsonr(y_test, y_test_pred)[0]

# 将结果存入excel表中
df_train = pd.DataFrame({'Actual': y_train, 'Predicted': y_train_pred})
df_val = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred})

with pd.ExcelWriter(r'D:\宋毅文\graduate student career\小论文\基于机器学习的化学链制氢载氧体的预测\机器学习训练\original model\Decision tree_results.xlsx') as writer:
    df_train.to_excel(writer, sheet_name='Training Set')
    df_val.to_excel(writer, sheet_name='Test Set')

# 输出结果
print("Cross-validation scores:", cv_scores)
print("Training set R2 score:", r2_train)
print("Validation set R2 score:", r2_test)
print("Training set RMSE score:", rmse_train)
print("Validation set RMSE score:", rmse_test)
print("Training set MAE score:", mae_train)
print("Validation set MAE score:", mae_test)
print("Training set MSE score:", mse_train)
print("Validation set MSE score:", mse_test)
print("Training set Pearson correlation coefficient:", pearson_train)
print("Validation set Pearson correlation coefficient:", pearson_test)
