import pandas as pd
import datetime
import numpy as np
df_station = pd.read_csv("2167869.csv")
df_station.sample(frac=1)
df_station[["temp", "qf"]] = df_station.TMP.str.split(",", expand = True) 
df_station[['day','time']] = df_station.DATE.str.split("T",expand=True)

df_station['day'] = df_station['day'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d").timetuple().tm_yday)
df_station['time'] = df_station['time'].apply(lambda x: x[0:2] + x[3:5])


given = 150
#given_days = np.arange(139, 176)
chunks = []
for station, dataframe in df_station.groupby(["STATION"]):
    chunks.append(dataframe)
    
    
station = []
stations = []
#for given in given_days: 
for chunk in chunks:
    for index, row in chunk.iterrows():
        if row["day"] == given:
            dummy = {}
            dummy["time"] = row["time"]
            dummy["temp"] = row["temp"]
            dummy["id"] = row["STATION"]
            dummy["day"] = row["day"]
            station.append(dummy)
    stations.append(station)
    station = []
    
    
print(stations)

df_video = pd.read_excel("video.xlsx")
video = []
#for given in given_days:
for i, value in enumerate(df_video["day"]):
    if df_video["day"][i] == given:
        dummy = {}
        dummy["time"] = df_video["time"][i]
        dummy["temp"] = df_video["temp(C)"][i]
        dummy["day"] = df_video["day"][i]
        video.append(dummy)

    



from scipy import interpolate
import numpy as np
x =[]
y =[]
interpolations = []
for station in stations:
    for i in station:
        x.append(int(i["time"]))
        y.append(int(i["temp"])/10)
    interpolations.append( [i["id"], interpolate.interp1d(x, y)])
    
    

xdash =[]
ydash =[]

for i in video:
    xdash.append(int(i["time"]))
    ydash.append(int(i["temp"]))

errors = []
from sklearn.metrics import mean_squared_error  
for interpol in interpolations:
    ypredicted = interpol[1](xdash)
    mse = mean_squared_error(ydash, ypredicted)
    errors.append([interpol[0], mse])
    
    
    
import matplotlib.pyplot as plt
plt.scatter(*zip(*errors))
plt.axis([71006099999,71986099999, 300, 400])
plt.show()




