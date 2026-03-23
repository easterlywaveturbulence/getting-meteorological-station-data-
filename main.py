#%%
import matplotlib.pyplot as plt
from datetime import datetime

from meteostat import Stations, Point, Daily,Hourly
import matplotlib.dates as mdates
import numpy as np
import statsmodels.api as sm

# 获取气象站点
stations = Stations()
stations1 = stations.nearby(30, 112)
station = stations1.fetch(50)
#选择距离北纬30，东经112度附近最近的50个气象站点
df = station[station['country'] == 'CN']
#只选择中国大陆地区站点
df.to_csv('./ooutput.csv', index=False)

print(station)
 #获取天气数据
start = datetime(2008, 1, 8,8,0,0)
end = datetime(2008, 2, 15,0,0,0)
#time1 = datetime(2025, 1, 18, 10, 0, 0)
#time2 = datetime(2025, 1, 18, 15, 0, 0)
location = Point(30.6167,114.1333,23)
#武汉站纬度，经度，海拔
data = Daily(location, start, end)
#逐日数据
data2 = data.fetch()

# 绘制条形图
ax = data2.plot(y=['tavg'], kind='bar', figsize=(12, 6), legend=False)

# 设置日期格式化和间隔
ax.xaxis.set_major_locator(plt.MaxNLocator(10))  # 共设置10个主刻度 
ax.set_xlabel("Date")  # 设置 x 轴标签
ax.set_ylabel("Average Temperature (°C)")  # 设置 y 轴标签
#ax.set_title("Average Daily Temperature in QiQiHaEr")  # 设置标题

# 调整布局
plt.xticks(rotation=45)  # 旋转 x 轴标签
plt.tight_layout()

# 显示图例
plt.legend(["Snow Depth"], loc='upper left')

# 显示图表
plt.show()
stations = Stations()
stat2=stations.nearby(20,111)
station2 = stat2.fetch(15)
df3 = station2[station2['region'] == 'HA']
a=df3.iat[0,5]
b=df3.iat[0,6]
c=df3.iat[0,7]
start2=datetime(2024,6,1,0,0)
end2=datetime(2024,12,1,0,0)

location2=Point(a,b,c)
shuju=Hourly(location2,start2,end2)
shuju2=shuju.fetch()
fig=plt.figure(figsize=(12,6))
ax2=fig.add_subplot()
shuju3=shuju2.iloc[:,0]
shuju3=shuju3.interpolate(method='time')
ax2.plot(shuju3)
shuju3 = shuju3.rolling(window=100, min_periods=1).mean()
shuju3=shuju3.interpolate(method='time')
ax2.xaxis.set_major_locator(plt.MaxNLocator(10)) 
ju=shuju2.sort_values('wspd',ascending=False)
ju2=shuju2[shuju2.index.month==6]
ju2=ju2[ju2.index.day==10]
ju3=ju.iloc[:,3]
a=ju3.sum()
boollist=[]
ll=[]
l=[]
lll=[]
print(ju.iat[3,8])
for i in range(4390):

    pre=shuju2.iat[i,8]
    pre3=shuju2.iat[i+3,8]
    l.append(pre)
    lll.append(pre3)
    inde=pre3-pre
    ll.append(inde)
    if abs(inde)>3:
        boollist.append(True)
    else:
        boollist.append(False)
boollist.append(False)
boollist.append(False)
boollist.append(False)
shujuwind=shuju2.loc[boollist]
xx = np.arange(len(shuju3))  # 时间序列作为 x
yy = shuju3.values          # 温度值作为 y

# 添加常数项（截距）
xx = sm.add_constant(xx)

# 使用 statsmodels 进行回归
model = sm.OLS(yy, xx)
results = model.fit()

# 打印回归分析结果
print(results.summary())

# 提取回归线
y_pred = results.predict(xx)
#print(results.summary())
# 绘图
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(shuju3.index, yy, label='Original Data', color='blue')
ax.plot(shuju3.index, y_pred, label='Linear Regression', color='red', linestyle='-')
ax.set_xlabel("Date")
ax.set_ylabel("Temperature")
ax.legend()
plt.show()
predictions = results.get_prediction(xx)
summary_frame = predictions.summary_frame(alpha=0.001)  # 99.9% 置信区间
y_pred = summary_frame["mean"]
low=summary_frame.iloc[:,4]
lower_bound = summary_frame["mean_ci_lower"]
upper_bound = summary_frame["mean_ci_upper"]

# 绘图
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(shuju3.index, yy, label="Observed Data", color="blue")
ax.plot(shuju3.index, y_pred, label="Fitted Line", color="red", linestyle="--")
ax.fill_between(shuju3.index, lower_bound, upper_bound, color="yellow", alpha=0.9, label="99.9% Confidence Interval")
ax.set_xlabel("Date")
print(shuju3.idxmax())
 
ax.annotate('Max value', xy=(shuju3.idxmax(),32), xytext=(shuju3.idxmax() ,29),
            arrowprops=dict(facecolor='red', shrink=0.05))
ax.set_ylabel("Temperature")
ax.legend()
plt.title("Temperature with 99.9% Confidence Interval")
plt.show()
