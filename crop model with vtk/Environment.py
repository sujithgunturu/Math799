# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 17:30:01 2020

@author: welchsm
"""
import pandas as pd
import datetime as DT
import math
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from sys import exit

# Read weather data and construct dictionary of interopolating polynomials
df = pd.read_excel('Weather_Ahwaz_International_Airport_Iran.xlsx')
WIntrp={k:CubicSpline(np.arange(1.,367.),df[k],bc_type='natural') 
        for k in df.keys()}

def GetValue(vName,day,year=2020):
    """
    

    Parameters
    ----------
    vName : TYPE
        DESCRIPTION.
    day : TYPE
        DESCRIPTION.
    year : TYPE, optional
        DESCRIPTION. The default is 2020.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    value : TYPE
        DESCRIPTION.

    """
    
    # If necessary, adjust day and year
    if day<1: raise Exception("Invalid day: %d"%(day))
    add_year=0
    while day>366:
        day-=366
        add_year+=1
    year+=add_year
    if not 1989<year<2031: raise Exception("Invalid year %d"%(year))
    if year%4!=0 and day>31+28: day-=1
    
    # Fetch requested value trapping a bad vName or date. Legal vName's are
    #           "sunlight"
    #           "lowtemp"
    #           "hightemp"
    #           "hrs_sun"
    #           "GDD_clim"
    #           "cloudy"    
    try:
        value=WIntrp[vName](day)
    except:
        raise Exception("Invalid weather request: %s %d %d"%(vName,day,year))
    
    # Convert units as needed
    if 'temp' in vName:             # Farenheit to Centigrade
        value=5*(value-32)/9
    if vName=='sunlight':           # kW-hrs/m^2/day to MJ/m^2/day
        value*=3.6
    
    # Return it
    return value

    
if __name__ == '__main__':
    
    Mat=GetValue('GDD_clim',365)-GetValue('GDD_clim',258)+GetValue('GDD_clim',28)
    print(Mat)
