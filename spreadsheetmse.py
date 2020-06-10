import pandas as pd
import datetime
df_video = pd.read_excel("video.xlsx")
given = 145
video = []
for i, value in enumerate(df_video["day(utc)"]):
    if df_video["day(utc)"][i] == given:
        dummy = {}
        dummy["time"] = int(df_video["time(utc)"][i])
        dummy["temp"] = df_video["temp(C)"][i]
        dummy["day"] = df_video["day(utc)"][i]
        video.append(dummy)     
from scipy import interpolate
import numpy as np
x =[]
y =[]
interpolations = []
for i in video:
    x.append(i["time"])
    y.append(i["temp"])
f = interpolate.interp1d(x, y)
p = 500
q = 1500
r = (q-p)//100 + 1
xdash =  np.linspace(p, q, r)
ypredicted = f(xdash)
predection_task = []
for a, b in zip(xdash, ypredicted):
    predection_task.append([a, b])
#predection_task = [list(xdash), list(ypredicted)]
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

chunks = []
for station, dataframe in df_station.groupby(["STATION"]):
    chunks.append(dataframe)
    
    
station = []
stations = []
for chunk in chunks:
    for index, row in chunk.iterrows():
        if row["day"] == given and int(row["time"]) >= min(xdash) and int(row["time"]) <= max(xdash):
            dummy = {}
            dummy["time"] = int(row["time"])
            dummy["temp"] = int(row["temp"])/10
            dummy["id"] = row["STATION"]
            dummy["day"] = row["day"]
            station.append(dummy)                
    stations.append(station)
    station = []
from sklearn.metrics import mean_squared_error  
from math import sqrt
errors = []
arrayformse = []
for station in stations:
    station = list({v['time']:v for v in station}.values())
    if len(station) == 0:
        continue
    if len(station) != len(xdash):
        ypred_temp = []
        for elem1 in station:
            for elem2 in predection_task:
                if elem1["time"] == elem2[0]:
                    ypred_temp.append(elem2[1])
                
             
    for i in range(len(station)):
        arrayformse.append(station[i]["temp"])
    try:
        if len(station) == len(xdash):
            mse = sqrt(mean_squared_error(arrayformse, ypredicted))
        elif len(station) > 0 and len(station)!=len(xdash):
            mse = sqrt(mean_squared_error(arrayformse, ypred_temp))
        else:
            mse = 0
    except:
        continue
    mseid = station[i]["id"]
  
    errors.append([mseid, mse])
    arrayformse = []


import matplotlib.pyplot as plt
plt.scatter(*zip(*errors))
plt.axis([71006099999,71986099999, 0, 100])
plt.show()   
    
    

from operator import itemgetter
sortedlist = sorted(errors, key=itemgetter(1))

toplotx = []
toploty = []
for i in stations[14]:
    toplotx.append(i["time"])
    toploty.append(i["temp"])
import matplotlib.pyplot as slt 
slt.plot(xdash,ypredicted,'r', label='interp/extrap')
slt.plot(x,y, 'b--', label='data')
slt.plot(toplotx, toploty , 'r', label= "station")
slt.legend()


#a = sortedlist[:]
#for i in range(len(a)):
#    del a[i][0]
#    a[i].insert(0, i)
#plt.scatter(*zip(*a))
#plt.axis([1,100, 0, 1000])
#plt.show()   


#toplotx = []
#toploty = []
#All = []
#for station in stations:
    #for i in stations:
    #    toplotx.append(i["time"])
    #    toploty.append(i["temp"])
    #    All.append([toplotx, toploty])
    #    toplotx =[]
    #    toploty = []
#import matplotlib.pyplot as slt 
#slt.plot(xdash,ypredicted,'r', label='interp/extrap')
#slt.plot(x,y, 'b--', label='data')
#for k in range(10)
    #slt.plot(*zip(*All[k]),'r')
    #slt.axis([0, 2400, 0, 30])
#slt.legend()
