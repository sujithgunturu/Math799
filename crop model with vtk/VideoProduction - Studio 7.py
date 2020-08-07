# -*- coding: utf-8 -*-
"""
Purpose: To make a video of a growing plant driven by a less simple wheat model.
         The major elements of the method are drawn from code found at
         https://theailearner.com/2018/10/15/creating-video-from-images-using-opencv-python/
         
         and from
         
         https://lorensen.github.io/VTKExamples/site/Python/IO/ImageWriter/
         
Created on Tue Apr 21 18:02:36 2020
@author: welchsm
"""
import vtk
import cv2
import numpy as np
import glob
import os
from math import *
from Environment import *
from Single_Wheat_Plant_Visualizer import *
from SpringWheat import *
from matplotlib.colors import hsv_to_rgb
from time import perf_counter
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sys import exit

def MakeVideo(v_name,F_del=False):
    
    # Obtain the list of images from the file, alphanumeric by file name 
    img_array = []; fnames=[]
    for filename in glob.glob('*.jpg'):
        fnames+=[filename]
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
     
    # Instantiate a video writer    
    out = cv2.VideoWriter(v_name+'.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    
    # Make the video
    for i in range(len(img_array)):
        out.write(img_array[i])
    
    # Release it
    out.release()
    
    # Delete image files if requested
    if F_del:
        for filename in fnames:
            os.remove(filename)
        

def WriteImage(fileName, renWin, rgba=True):
    """
    Write the render window view to an image file.

    Image types supported are:
     BMP, JPEG, PNM, PNG, PostScript, TIFF.
    The default parameters are used for all writers, change as needed.

    :param fileName: The file name, if no extension then PNG is assumed.
    :param renWin: The render window.
    :param rgba: Used to set the buffer type.
    :return:
    """
    if fileName:
        # Select the writer to use.
        path, ext = os.path.splitext(fileName)
        ext = ext.lower()
        if not ext:
            ext = '.png'
            fileName = fileName + ext
        if ext == '.bmp':
            writer = vtk.vtkBMPWriter()
        elif ext == '.jpg':
            writer = vtk.vtkJPEGWriter()
        elif ext == '.pnm':
            writer = vtk.vtkPNMWriter()
        elif ext == '.ps':
            if rgba:
                rgba = False
            writer = vtk.vtkPostScriptWriter()
        elif ext == '.tiff':
            writer = vtk.vtkTIFFWriter()
        else:
            writer = vtk.vtkPNGWriter()

        windowto_image_filter = vtk.vtkWindowToImageFilter()
        windowto_image_filter.SetInput(renWin)
        windowto_image_filter.SetScale(1)  # image quality
        if rgba:
            windowto_image_filter.SetInputBufferTypeToRGBA()
        else:
            windowto_image_filter.SetInputBufferTypeToRGB()
            # Read from the front buffer.
            windowto_image_filter.ReadFrontBufferOff()
            windowto_image_filter.Update()

        writer.SetFileName(fileName)
        writer.SetInputConnection(windowto_image_filter.GetOutputPort())
        writer.Write()
    else:
        raise RuntimeError('Need a filename.')
        
        
# Make one frame.  The arguments are:
#   Wheat   - Wheat plant model instance
#   VarD    - Dictionary of time series &/o interpolators from Wheat model run
#   day     - "Day" for which a frame is needed (can be a fraction)
def One_Wheat_Frame(Wheat,VarD,day,Axis=True):
    
    # Obtain the interpolated values of GDDAP, Idv, and Biomass
    GDDAP=VarD['GDDAP'](day); Idv=VarD['Idv'](day); Biomass=VarD['Biomass'](day)
    
    # Use them to obtain the Leaf dictionaries and internode lengths
    LeafDicts,Internodes=Wheat.Morphology(GDDAP,Idv,Biomass)

    # Draw the plant, returning the picture
    return Draw_Plant(LeafDicts,Internodes,Axis=Axis,Intr=False)
    
""" Create the movie frame by frame """
def main(SowDate):

    # Instantiate and run wheat model, and get interpolators for Growing 
    # Degree days After Planting, Integrated Developmental Units, and 
    # Biomass (all functions of Days After Planting), along with raw 
    # time series points for several other variables.
    Star=Wheat_S(SowDate)
    VarD,Units=Star.Model_interface()
    
    # Initialize movie descriptors
    Length=35                       # Time in seconds
    Frames_per_second=15            # Frames per second
    Frame_name="Whee_"              # Each frame is "Whee_"+num+".jpg
    Movie_Title="Growing_wheat"     # Avi file name
                                    # Number of frames to make
    Last_frame=Length*Frames_per_second
    
    # Function to convert frame number to Days After Planting
    Fn2d=lambda fn: (fn/Last_frame)*VarD['DAP'][-1]
    
    # Make each frame - do last one first to get largest bounding box. This
    # is necessary because (1) the leaves are positioned within the box and
    # (2) the spacing of the leaves controls the camera position, at least
    # as currently implemented.
    Stills=[*range(0,Last_frame+1)]
    Stills[-1],Stills[0]=Stills[0],Stills[-1]
    for fn in Stills:
        
        # Make a wheat plant image...
        renWin=One_Wheat_Frame(Star,VarD,Fn2d(fn),Axis=False)
        
        # Write it out as a jpeg...
        WriteImage(Frame_name+"%03d.jpg"%(fn), renWin, rgba=False)
         
    """ Make the video !! """
    MakeVideo(Movie_Title)

if __name__ == '__main__':

    # Pick the planting date & what plots are wanted then run the model
    Beg=perf_counter()
    SowDate=DT.datetime(2003,9,15)
    main(SowDate)
    print("Time required",perf_counter()-Beg)            
