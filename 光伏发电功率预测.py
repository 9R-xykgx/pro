import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 假设有历史数据，包含日照时长和对应的光伏发电功率
data = pd.read_csv('solar_power_data.csv')  # 数据文件包含'daylight_hours'和'power_output'列
X = data[['daylight_hours']]
y = data['power_output']

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 模型训练
model = LinearRegression()
model.fit(X_train, y_train)

# 预测
predictions = model.predict(X_test)

# 可视化结果
plt.scatter(X_test, y_test, color='blue', label='实际值')
plt.scatter(X_test, predictions, color='red', label='预测值')
plt.xlabel('日照时长 (小时)')
plt.ylabel('光伏发电功率 (kW)')
plt.legend()
plt.title('光伏发电功率预测')
plt.show()