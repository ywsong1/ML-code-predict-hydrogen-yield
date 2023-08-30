import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 读取数据集
data = pd.read_csv('test.CSV')

# 分离自变量和因变量
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 数据归一化处理
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 随机划分训练集和验证集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用XGBoost回归模型进行训练和预测
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1,
                             learning_rate=0.05, max_depth=6, n_estimators=1000)
xgb_model.fit(X_train, y_train)

# 交叉验证
scores = cross_val_score(xgb_model, X, y, cv=5, scoring='r2')
print('Cross validation R2 scores:', scores)
print('Mean R2 score:', np.mean(scores))

# 在训练集和验证集上进行预测和评估
y_train_pred = xgb_model.predict(X_train)
y_val_pred = xgb_model.predict(X_val)

print('Training set:')
print('R2 score:', r2_score(y_train, y_train_pred))
print('RMSE:', np.sqrt(mean_squared_error(y_train, y_train_pred)))
print('MAE:', mean_absolute_error(y_train, y_train_pred))
print('MSE:', mean_squared_error(y_train, y_train_pred))
print('Pearson correlation coefficient:', np.corrcoef(y_train, y_train_pred)[0][1])

print('Validation set:')
print('R2 score:', r2_score(y_val, y_val_pred))
print('RMSE:', np.sqrt(mean_squared_error(y_val, y_val_pred)))
print('MAE:', mean_absolute_error(y_val, y_val_pred))
print('MSE:', mean_squared_error(y_val, y_val_pred))
print('Pearson correlation coefficient:', np.corrcoef(y_val, y_val_pred)[0][1])

# 将结果存入excel表中
df_train = pd.DataFrame({'Actual': y_train, 'Predicted': y_train_pred})
df_val = pd.DataFrame({'Actual': y_val, 'Predicted': y_val_pred})

with pd.ExcelWriter(r'D:\宋毅文\graduate student career\小论文\基于机器学习的化学链制氢载氧体的预测\机器学习训练\orginal results\XG Boost_results.xlsx') as writer:
    df_train.to_excel(writer, sheet_name='Training Set')
    df_val.to_excel(writer, sheet_name='Test Set')