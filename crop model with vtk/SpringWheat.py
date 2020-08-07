# -*- coding: utf-8 -*-
"""
Purpose - Implements a slightly adapted model the spring wheat
          variety 'Star'. The model is described in the paper 
          Andarzian B, et al. 2008. Biosystems Engineering 99:487-95.
Created on Thu Apr 23 12:47:14 2020
@author: welchsm
"""
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from copy import deepcopy
from Environment import *
from math import *
import numpy as np
import pandas as pd
import datetime as DT
from sys import exit

""" Get weather data """        
# Latitude & longitude of Ahvaz Res. Sta., Iran, in radians
Lat=(31+21/60)*pi/180              
Lon=(48+ 8/60)*pi/180 
             
# Read weather data and construct dictionary of interopolating polynomials
df = pd.read_excel('Weather_Ahwaz_International_Airport_Iran.xlsx')
WIntrp={k:CubicSpline(np.arange(1.,367.),df[k],bc_type='natural') 
        for k in df.keys()}

# Define weather access functions
SunFrac=lambda d: GetValue('cloudy',d)              # Cloud transmission
MinTemp=lambda d: GetValue('lowtemp',d)             # Daily min temperature
MaxTemp=lambda d: GetValue('hightemp',d)            # Daily max temperature
HrsLite=lambda d: GetValue('hrs_sun',d)             # Daily photoperiod
DegDays=lambda d: GetValue('GDD_clim',d)            # Growing degree days based
                                                    #   on climate data (base 50 F,
                                                    #   upper thresh 86 F)
SunLite=lambda d: GetValue('sunlight',d)            # Daily solar radiation
GDDpDay=lambda d: GetValue('DailyGDD',d)            # Daily degree day

# Photosynthetically active radiation.  The scale factor is from
# https://en.wikipedia.org/wiki/Photosynthetically_active_radiation 
PAR    =lambda d: GetValue('sunlight', d)*0.368     # MJ/m^2/d

# Temperature interconversion functions
degC2F=lambda C: 9*C/5+32
degF2C=lambda F: 5*(F-32)/9

# Growing degree-days function using the method described at
# https://ndawn.ndsu.nodak.edu/help-wheat-growing-degree-days.html
def GDD(Tmin,Tmax,Tbase=0,Thresh=35,deg='C'):
    if deg not in ['C','F']: raise Exception("Invalid deg system requested")
    if deg=='F':
        Tmin=degC2F(Tmin)
        Tmax=degC2F(Tmax)
        Tbase=degC2F(Tbase)
        Thresh=degC2F(Thresh)
    if Tmin>Tmax  : raise Exception("Tmin cannot be greater than Tmax")
    if Tmin>Thresh: return Thresh
    if Tmax<Tbase : return 0.
    if Tmax>Thresh: Tmax=Thresh
    if Tmin<Tbase : Tmin=Tbase
    return (Tmin+Tmax)/2
   
""" Biological curves """
# General developmental rate functions
alpha=lambda tmin,topt,tmax: log(2)/log((tmax-tmin)/(topt-tmin))
def TempDevRate(T,Tmin,Topt,Tmax):
    a=alpha(Tmin,Topt,Tmax)
    if Tmin<T<Tmax :
        return (2*(T-Tmin)**a*(Topt-Tmin)**a-(T-Tmin)**(2*a))/(Topt-Tmin)**(2*a)
    return 0.

# Vegetative developmental response to temperature
def VegTempDevRate(T,Tmin=0,Topt=24,Tmax=35):
    return TempDevRate(T,Tmin,Topt,Tmax)
    
# Vegetative developmental response to temperature
def RprTempDevRate(T,Tmin=8,Topt=29,Tmax=40):
    return TempDevRate(T,Tmin,Topt,Tmax)

# Light interception as a function of development stage
def DevIndxRadIncp(Idv,a=0.943,b=0.480,c=0.116):
    return a/(1+exp((b-Idv)/c))

""" Spring wheat model """
class Wheat_S(object):
    
    # Initialize model
    def __init__(self,SowDate):
        
        # Get DOY for sowing date
        year=SowDate.year
        self.SowDate=(SowDate-DT.datetime(year,1,1)).days+1
        
        # Get DOY for emergence date by accumulating degree-days. Assume
        # this takes 180 GDD degF. Note: Degree days in weather file used.
        self.DAP=0
        self.GDDAP=0
        while self.GDDAP<=180:
            day=self.SowDate+self.DAP
            self.GDDAP+=GDDpDay(day)
            self.DAP+=1
        self.EmgDate=self.SowDate+self.DAP
        
        # Set development stage and biomass
        self.Idv=0
        self.Biomass=0
        
        # Set wheat variety parameters based on Star
        self.VegStage=1400      # Vegetative stage growing degree days
        self.RprStage=1017      # Reproductive stage growing degree days
        self.RUE=3              # Radiation use efficiency g/MJ
        self.HI=0.36            # Harvest index
        
        # The paper has a factor in the yield equation called delta_Ri 
        # that is defined as "fraction of the maximum RUE depending on crop
        # performance".  (It is assigned units of g/MJ^2 but this is badly
        # in error - it is actually unitless.)  This is clearly a fudge
        # factor that is discussed nowhere else in the paper.  This author
        # calculated the following value to make the predicted final biomass
        # agree with the obseverd value.
        self.delta_Ri=0.887
        
        # The following values are fudged to make the reported days to
        # maturity agree with the model predictions.  This factor is
        # probably needed because of some unreported model subtlety
        FF=2.0                  # The fudge factor (aka "calibration")
                                # This value makes one suspect a mixup
                                #   between Farenheit and Centigrade
        self.Dmax_v=0.0115*FF   # Maximum vegetative development rate
        self.Dmax_r=0.0244*FF   # Maximum reproductive development rate
        
    # Simulate one day
    def One_Day(self,day):
        
        # Retrieve environmental data
        SF=SunFrac(day)             # Cloud transmission
        Tmin=MinTemp(day)           # Daily min temperature
        Tmax=MaxTemp(day)           # Daily max temperature
        DayLength=HrsLite(day)      # Hours of sunlight
        
        P_A_R=PAR(day)              # Photosyn. Act. Rad. for today
        if self.Idv<=1: Th=23.0
        else: Th=35.                # Growing degree days for today
        G_D_D=GDD(Tmin,Tmax,Thresh=Th)        
        T=(Tmin+Tmax)/2             # Average temperature for today
        
        # Compute biological rates
        if self.Idv<1.0:            
            # Vegetative stage dev. rate
            f_Temp=VegTempDevRate(T)
            D_iv=self.Dmax_v*f_Temp
            
        else:
            # Reproductive stage dev. rate
            f_Temp=RprTempDevRate(T)
            D_iv=self.Dmax_r*f_Temp
            
        # Biomass increment
        f_i=DevIndxRadIncp(self.Idv)# Fraction intercepted PAR
        dBiomass=self.RUE*f_i*P_A_R*self.delta_Ri
        
        # Advance running totals
        self.DAP+=1                 # Days after planting
        self.GDDAP+=G_D_D           # GDD after planting
        self.Idv+=D_iv              # Development index
        self.Biomass+=dBiomass      # Plant biomass
        
        # Done - return some daily quantities of interest
        return SF,DayLength,T,P_A_R,G_D_D,f_Temp,f_i
    
    # Run model from planting to Idv=2
    def RunModel(self):
        
        # Initialize plotting stuff
        State_Vars=[]
        Daily_Vars=[]
        
        # Iterate over days, saving both state and daily variables
        while self.Idv<=1:
            
            # Run model for one day
            day=(self.SowDate+self.DAP)
            DayStuff=self.One_Day(day)
            
            # Save variables
            Daily_Vars+=[[*DayStuff]]
            State_Vars+=[[self.DAP,self.GDDAP,self.Idv,self.Biomass]]

        # Done - return stuff to plot
        return Daily_Vars,State_Vars
    
    # Interface routine that runs the model and returns a dictionary with 
    # cubic spline output variable interoplators for specified variables.
    # A second dictionary returns units of all variables.
    def Model_interface(self,Cubic=['GDDAP','Idv','Biomass']):
        
        # Run the model thru harvest
        Daily_Vars,State_Vars=self.RunModel()
        
        # Unpack quantities into columns
        Daily_Vars,State_Vars=np.array(Daily_Vars),np.array(State_Vars)
        DAP,GDDAP=State_Vars[:,0],State_Vars[:,1]
        Idv,Biomass=State_Vars[:,2],State_Vars[:,3]
        
        SF,DayLength=Daily_Vars[:,0],Daily_Vars[:,1]
        T,P_A_R=Daily_Vars[:,2],Daily_Vars[:,3]
        G_D_D,f_Temp=Daily_Vars[:,4],Daily_Vars[:,5]
        f_i=Daily_Vars[:,6]

        # Make dictionaries of variables and units
        Stuff={'DAP':DAP,'GDDAP':GDDAP,
               'Idv':Idv,'Biomass':Biomass*10,
               'SF':SF,'DayLength':DayLength,'T':T,
               'PAR':P_A_R,'GDD':G_D_D,'f_Temp':f_Temp,
               'f_i':f_i}
        
        Units={'DAP':'(days after planting)',
               'GDDAP':'(GDD after planting)',
               'Idv':'(Development units)','Biomass':'(Biomass kg/ha)',
               'SF':'(unitless)',
               'DayLength':'(Hrs)','T':'(deg C)','PAR':'(MJ/m^2/d)',
               'GDD':'(deg F - day)','f_Temp':'(unitless)',
               'f_i':'(unitless)'}
        
        # Convert results vectors into cubic splines
        for key in Stuff.keys():
            if not key in Cubic: continue
            Stuff[key]=CubicSpline(Stuff['DAP'],Stuff[key],bc_type='natural')
            
        # Done
        return Stuff,Units
    
    # Converts GDDAP, Idv, and Biomass data into a list of leaf Dictionaries
    # and Internode lengths.  
    # NOTE: CURRENTLY IMPLEMENTED FOR VEGETATIVE GROWTH ONLY
    def Morphology(self,GDDAP,Idv,Biomass):
        
        # Fake the phenology
        Pchron=171.              # Totally ad hoch phyllochron interval
        maxGDDAP=1283.           # GDDAP at Idv=1
        Haun=GDDAP/Pchron        # Hokey Haun stage
        FHaun=Haun-floor(Haun)   # Fractional Haun state
        NLeaf=ceil(Haun)         # Number of leaves to make
        
        # Fake the growth 
        maxBiomass=4120          # Maximum biomass
        
        # Initialize the leaf dictionaries with invarient values
        D={
             'wmax': 0.05,       #  Rel max width position [0.0375,0.0675]
            'swmax': .2,         #  Rel max width [0.1,0.4] 
               'f1': .60,        #  First form factor  [0.55,0.70]
               'f2': .85,        #  Second form factor [0.75,0.90]
             'roll': 0.,         #  Ligule roll angle [-pi/9,+pi/9] 
            'pitch': 0.,         #  Ligule pitch angle [0] (The8a1 governs)
              'yaw': 0.,         #  Ligule yaw angle as determined by PhyT
           'ligule': None,       #  Ligule position 
            }
        
        LeafDicts=[deepcopy(D) for i in range(NLeaf)]
        
        # Calculate allocation fractions for each leaf
        L=[x if x<Haun else FHaun 
           for x in [float(i+1) 
                     for i in range(NLeaf)]]
        sumL=sum(L)
        Factors=np.array([(NLeaf-i)/sumL for i in range(1,NLeaf)]+[L[-1]/sumL])
        Factor2=Factors**(1/2); Factor2/=np.sum(Factor2)
        Factor3=Factors**(1/3); Factor3/=np.sum(Factor3)
        
        # Assign current and maximum phytomer (leaf+internode) fractional growth
        PhytomerGroFrac=Biomass*Factor2/maxBiomass/Factor2[0]

        # Add other entries to leaf dictionaries
        for i,D in enumerate(LeafDicts[0:len(Factors)]):  
                                        # Midrib length [2.,40.]
            D['lmax']=2+PhytomerGroFrac[i]*38
                                        #  First launch angle  [3pi/12,5pi/12]
            #D['Th8a1']=1*pi/12+(1-PhytomerGroFrac[i])*4*pi/12
            D['Th8a1']=4*pi/12
                                        #  Second launch angle [2pi/12,4pi/12]
            #D['Th8a2']=2*pi/12+   PhytomerGroFrac[i]*3*pi/12
            D['Th8a2']=5.5*pi/12
            D['Leaf']=i+1               #  Leaf number

        # Determine internode lengths
        PhytomerGroFrac=Idv*Factor3/Factor3[0]
        Internodes=list(10*PhytomerGroFrac)
    
        # Done
        return LeafDicts,Internodes
        
# Run model and plot stuff
def main(SowDate):

    # Instantiate model with the Star cultivar parameters given planting date
    Star=Wheat_S(SowDate) 

    # Run interfaced model
    Cubic=['GDDAP','Idv','Biomass']
    Stuff,Units=Star.Model_interface(Cubic=Cubic)
    
    # Plot everything against DAP        
    days=Stuff['DAP']
    for key in Stuff.keys():
        if key=='DAP': continue
        if key in Cubic:
            plt.plot(days,Stuff[key](days))
        else:
            plt.plot(days,Stuff[key])
        plt.xlabel('DAP'+' '+Units['DAP'])
        plt.ylabel( key+' ' +Units[key])
        plt.show()
    print(Stuff['Biomass'](days[-1]))
# Run program
if __name__ == '__main__':
    
    # Pick the planting date & what plots are wanted then run the model
    SowDate=DT.datetime(2003,9,15)
    main(SowDate)            
        