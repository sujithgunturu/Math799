import pandas as pd
import datetime
df_video = pd.read_excel("video.xlsx")
given = 150
video = []
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
for i in video:
    x.append(i["time"])
    y.append(i["temp"])
f = interpolate.interp1d(x, y, fill_value="extrapolate")
xdash =  np.linspace(0000, 2400, 24)
ypredicted = f(xdash)
predection_task = [list(xdash), list(ypredicted)]
from xlwt import Workbook 
wb = Workbook()
sheet1 = wb.add_sheet('Sheet 1')
j = 0
for temps, data in zip(xdash, ypredicted):
    
    sheet1.write(j, 0, temps)
    sheet1.write(j, 1, data)
    j += 1
wb.save('inter.xls')

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
        if row["day"] == given and int(row["time"]) >= 0000 and int(row["time"]) <= 2300:
            dummy = {}
            dummy["time"] = int(row["time"])
            dummy["temp"] = int(row["temp"])/10
            dummy["id"] = row["STATION"]
            dummy["day"] = row["day"]
            station.append(dummy)
    #if station == []:
    #    dummy = {}
    #    dummy["id"] = row["STATION"]
    #    station.append(dummy)
                
    stations.append(station)
    station = []
from sklearn.metrics import mean_squared_error  
errors = []
arrayformse = []
for station in stations:
    station = list({v['time']:v for v in station}.values())
    if len(station) == 0:
        continue
    if len(station) != 24:
        ypred_temp = []
        for elem in station:
            if elem["time"] in predection_task[0]:
                ypred_temp.append(predection_task[1][predection_task[0].index(elem["time"])])
             
    for i in range(len(station)):
        arrayformse.append(station[i]["temp"])
    if len(station) == 24:
        mse = mean_squared_error(arrayformse, ypredicted)
    elif len(station) > 0 and len(station)!=24:
        mse = mean_squared_error(arrayformse, ypred_temp)
    else:
        mse = -1
    mseid = station[i]["id"]
  
    errors.append([mseid, mse])
    arrayformse = []


import matplotlib.pyplot as plt
plt.scatter(*zip(*errors))
plt.axis([71006099999,71986099999, 0, 1000])
plt.show()   
    
    

from operator import itemgetter
sortedlist = sorted(errors, key=itemgetter(1))