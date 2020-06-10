import pandas as pd

import datetime

df_station = pd.read_csv("71006099999.csv")

df_station[["temp", "qf"]] = df_station.TMP.str.split(",", expand = True) 
df_station[['day','time']] = df_station.DATE.str.split("T",expand=True)

df_station['day'] = df_station['day'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d").timetuple().tm_yday)
df_station['time'] = df_station['time'].apply(lambda x: x[0:2] + x[3:5])
df_video = pd.read_excel("video.xlsx")

given = 150
station = []
video = []

for i, value in enumerate(df_station["day"]):
    if df_station["day"][i] == given:
        dummy = {}
        dummy["time"] = df_station["time"][i]
        dummy["temp"] = df_station["temp"][i]
        station.append(dummy)
                    
for i, value in enumerate(df_video["day"]):
    if df_video["day"][i] == given:
        dummy = {}
        dummy["time"] = df_video["time"][i]
        dummy["temp"] = df_video["temp(C)"][i]
        video.append(dummy)
        
        
        

import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np
x =[]
y =[]
for i in station:
    x.append(int(i["time"]))
    y.append(int(i["temp"])/10)
f = interpolate.interp1d(x, y)







xdash =[]
ydash =[]

for i in video:
    xdash.append(int(i["time"]))
    ydash.append(int(i["temp"]))

ypredicted = f(xdash)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(ydash, ypredicted)
print(mse)




fig = plt.figure()
ax1 = fig.add_subplot(111)


ax1.scatter(x, y, s=10, c='b', marker="s", label='station')
ax1.scatter(xdash,ydash,  s=10, c='r', marker="o", label='video')
plt.legend(loc='upper left');
plt.show()



