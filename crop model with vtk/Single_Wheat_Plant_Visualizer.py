# -*- coding: utf-8 -*-
"""
Purpose: To make one wheat plant plant based on a crop simulation model output

Created on Sun May  3 11:00:31 2020
@author: welchsm
"""
from math import *
import numpy as np
from numpy.linalg import norm
from numpy.random import uniform,seed,randint
from scipy.optimize import bisect
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from time import perf_counter
import vtk
from sys import exit

# Box for axes if they are selected
BigBox=None

# Array reverse function
reverse=lambda Ar: Ar[-1::-1]

# Sign and even/odd functions
sign=lambda x: 0 if x==0 else x/abs(x)
even=lambda x: not bool(x%2)

# Translation vectors as calculated in first frame
TrL8tr=None; FirstSpacing=True

# Generate a rotation matrix
def Rotator(roll=0.,pitch=0.,yaw=0.):
    
    # Convert decriptions to axis names
    x=roll; y=pitch; z=yaw
    
    # Compute individual axis rotations
    Rx=np.array([[1.,     0.,     0.],
                 [0., cos(x),-sin(x)],
                 [0., sin(x), cos(x)]])

    Ry=np.array([[ cos(y),0., sin(y)],
                 [0.,     1.,     0.],
                 [-sin(y),0., cos(y)]])

    Rz=np.array([[ cos(z),-sin(z),0.],
                 [ sin(z), cos(z),0.],
                 [     0.,     0.,1.]])

    # Compute final rotation matrix
    R=np.dot(Rz,np.dot(Ry,Rx))
    return R

# VTK pipeline device
class VTK_Pipe(object):
    
    # Set up device to render a list of leaf objects
    def __init__(self,verbose=False,Intr=True):

        # Enable/disable print and display functions
        self.verbose=verbose
        self.Intr=Intr
        
        # Make all objects that do not have prerequisites
        self.appendFilter = vtk.vtkAppendPolyData()
        self.cleanFilter = vtk.vtkCleanPolyData()
        self.mapper = vtk.vtkPolyDataMapper()
        self.actor = vtk.vtkActor()
        self.renderer = vtk.vtkRenderer()
        self.renderWindow = vtk.vtkRenderWindow()
        self.renderWindow.SetSize(700,700)
        if Intr:
            self.renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        
        # Pipeline set up
        self.mapper.SetInputConnection(self.cleanFilter.GetOutputPort())
        self.actor.SetMapper(self.mapper)
        self.renderWindow.AddRenderer(self.renderer)
        if Intr:
            self.renderWindowInteractor.SetRenderWindow(self.renderWindow)
        self.renderer.AddActor(self.actor)
        
        # Set miscellaneous properties
        self.actor.GetProperty().SetColor(0.00,.75,0.00) # Green leaves
        self.actor.GetProperty().SetAmbient(0)           # No ambient light
        self.actor.GetProperty().SetDiffuse(1)           # Only diffuse light
        self.renderer.SetBackground(0.00,0.00,0.00)      # Black background 
        
        # Done
        return

    # Make a set of rods
    def MakeRodSet(self,BegPts,EndPts,Radii,Color):
        
        # How many rods are there in the rodset?
        nRods=BegPts.shape[1]  # BegPts are column vectors
        
        # 'Fasces' is Latin for 'a bundle of rods'
        self.Fasces=[ (vtk.vtkLineSource(),     # [0] Make linesource
                       vtk.vtkPolyDataMapper(), # [1] Make line mapper
                       vtk.vtkActor(),          # [2] Make line actor
                       
                       vtk.vtkTubeFilter(),     # [3] Make tubefilter
                       vtk.vtkPolyDataMapper(), # [4] Make tube mapper
                       vtk.vtkActor())          # [5] Make tube actor                      
                       for i in range(nRods) ]
        
        # Set line beginings, endpoints, connect to tubes, set radii, cap,
        # connect lines and tubes to mappers, mappers to actors, set color and 
        # actors to renderer
        for i,rd in enumerate(self.Fasces):
            rd[0].SetPoint1(BegPts[:,i])                    # Set line begin
            rd[0].SetPoint2(EndPts[:,i])                    # Set line end
            rd[1].SetInputConnection(rd[0].GetOutputPort()) # Line mapper <== line
            rd[2].SetMapper(rd[1])                          # Line acctor <== mapper
            rd[3].SetInputConnection(rd[0].GetOutputPort()) # Tube <== Line
            rd[3].SetRadius(Radii[i])                       # Set tube radius
            rd[3].SetNumberOfSides(20)                      # Set number tube sides
            rd[3].CappingOn()                               # Add tube caps
            rd[4].SetInputConnection(rd[3].GetOutputPort()) # Tube mapper <== tube
            rd[5].SetMapper(rd[4])                          # Tube acctor <== mapper
            rd[5].GetProperty().SetColor(*Color)            # Set tube color
            self.renderer.AddActor(rd[2])                   # Renderer <== line
            self.renderer.AddActor(rd[5])                   # Renderer <== tube
        
        # Done
        return
    
    # Make a set of ellipsoids
    def MakeEllipsoidSet(self,Centers,Radii,MajorAxis,Colors):
        
        # How many ellipsoids are in the set?
        nElls=Centers.shape[1]              # Centers are column vectors
        
        # Are they varicolored or all the same?
        if len(Colors.shape)==1:
            Kolors=np.array(nElls*[Colors]).T
        else:
            Kolors=Colors
        
        # Futbol is Spanish for football
        self.Futbol=[ (vtk.vtkParametricEllipsoid(),
                       vtk.vtkParametricFunctionSource(),
                       vtk.vtkTransform(),
                       vtk.vtkTransformPolyDataFilter(),
                       vtk.vtkPolyDataMapper(),
                       vtk.vtkActor())
                      for i in range(nElls)]
        
        # Create the pipeline for each ellipsoid
        for i in range(nElls):
            self.Futbol[i][1].SetParametricFunction(self.Futbol[i][0])
            self.Futbol[i][3].SetTransform(self.Futbol[i][2])
            self.Futbol[i][3].SetInputConnection(
                                    self.Futbol[i][1].GetOutputPort())
            self.Futbol[i][4].SetInputConnection(
                                    self.Futbol[i][3].GetOutputPort())
            self.Futbol[i][5].SetMapper(self.Futbol[i][4])
            
        # Set the plotting resolution for each of the parametric functions
        for i in range(nElls):
            self.Futbol[i][1].SetUResolution(61)
            self.Futbol[i][1].SetVResolution(61)
            self.Futbol[i][1].SetWResolution(61)
            self.Futbol[i][1].Update()
        
        # Set the radii for each axis and obtain the current and desired
        # semimajor axis unit vectors
        SemiMajor=np.zeros(Radii.shape)
        GoalMajor=np.zeros(Radii.shape)
        Rot8Angle=np.zeros(Radii.shape[1])
        for i in range(nElls):
            
            # Set ellipsoid radii
            self.Futbol[i][0].SetXRadius(Radii[0,i])
            self.Futbol[i][0].SetYRadius(Radii[1,i])
            self.Futbol[i][0].SetZRadius(Radii[2,i])
            
            # Get the current and desired semimajor axis unit vectors
            SemiMajor[np.argmax(Radii[:,i]),i]=1.
            GoalMajor[:,i]=MajorAxis[:,i]/norm(MajorAxis[:,i])
            
            # Get the rotation angle between them
            Rot8Angle[i]=degrees(acos(np.dot(SemiMajor[:,i],GoalMajor[:,i])))
            
        # Get vectors about which the ellipsoid axis is to be rotated using
        # the right hand rule. 
        Rotation=np.cross(SemiMajor,GoalMajor,axisa=0,axisb=0,axisc=0)
        
        # Orient/position each ellipsoid, set its color, & connect to the renderer
        for i in range(nElls):
            
            # Position and orient each ellipsoid
            self.Futbol[i][2].RotateWXYZ(Rot8Angle[i],*Rotation[:,i])
            self.Futbol[i][2].Translate(*Centers[:,i])
            self.Futbol[i][3].Update()
            
            # Set its color
            self.Futbol[i][5].GetProperty().SetColor(*Kolors[:,i])
            
            # Connect the results to the renderer
            self.renderer.AddActor(self.Futbol[i][5])
                       
        # Done
        return
        
    
    # Make a strip
    def MakeStrip(self,Base,Edge):
        
        # Unpack Edge and Base lines
        BX=Base[0,:]; BY=Base[1,:]; BZ=Base[2,:]
        EX=Edge[0,:]; EY=Edge[1,:]; EZ=Edge[2,:]
        
        # Make a points object and insert the points
        points=vtk.vtkPoints()
        LenX=len(BX)
        numPts=2*LenX
        
        for i in range(LenX):
            points.InsertPoint(2*i,  BX[i],BY[i],BZ[i])
            points.InsertPoint(2*i+1,EX[i],EY[i],EZ[i])
        
        # Make a cell array 
        cellar=vtk.vtkCellArray()
        cellar.InsertNextCell(numPts)
        for i in range(numPts):
            cellar.InsertCellPoint(i)
            
        # Produce the strip
        Strip=vtk.vtkPolyData()
        Strip.SetPoints(points)
        Strip.SetStrips(cellar)
        
        # Done
        return Strip
        
    # Merge and clean a list of vtkPolyData objects
    def Merge_and_Clean(self,PD_List):
        
        # Merge PolyData objects
        for pd in PD_List:
            self.appendFilter.AddInputData(pd)
        self.appendFilter.Update()

        # Remove duplicate points        
        self.cleanFilter.SetInputConnection(self.appendFilter.GetOutputPort())
        self.cleanFilter.Update()
        
        # Done
        return
        
    # Merge a set of cleaner outputs
    def Clean_Some_Cleaners(self,CLNR_List):
        
        # Extract & append vtkPolyData objects
        for clnr in CLNR_List:
            pd=clnr.GetOutput() 
            self.appendFilter.AddInputData(pd)
        self.appendFilter.Update()
        
        # Remove duplicate points        
        self.cleanFilter.SetInputConnection(self.appendFilter.GetOutputPort())
        self.cleanFilter.Update()
        
        # Done
        return
    
    # Produce graphics
    def Fire_for_Effect(self):
        
        """ Make it HAPPEN! """
        
        # Render the image, timing how long it takes
        Beg=perf_counter()
        self.renderWindow.Render()
        End=perf_counter()
        
        # If this application wants an interactive output, do so
        if self.Intr:
            self.renderWindowInteractor.Start()
            
        # ...othewise, return the rendered image
        else:
            return self.renderWindow
        
        # If the user wants to see how long rendering took, print it
        if self.verbose:
            print("Render time = ",(End-Beg),"sec")
        
        # Done
        return

# Single leaf object
class Leaf(object):
    
    # Set up a leaf of random length, launch angle, max width,
    # and point of max width
    def __init__(self,
                 Morph,        # Dictionary of the leaf morphology; keys are:
                               #    'lmax': Midrib length [2.,40.]
                               #    'wmax': Rel max width position [0.0375,0.0675]
                               #   'swmax': Rel max width [0.1,0.4] 
                               #      'f1': First form factor  [0.55,0.70]
                               #      'f2': Second form factor [0.75,0.90]
                               #   'Th8a1': First launch angle  [3pi/12,5pi/12]
                               #   'Th8a2': Second launch angle [2pi/12,4pi/12]
                               #    'roll': Ligule roll angle [-pi/9,+pi/9] 
                               #   'pitch': Ligule pitch angle [0] (The8a1 governs)
                               #     'yaw': Ligule yaw angle as determined below
                               #            by PhyT
                               #  'ligule': Ligule position
                               #    'Leaf': Leaf number
                 Sx=1.,        # Area scaling factor for length & Width
                 pt_space=0.3, # Size of length segments
                 PhyT=True,    # Use wheat phyllotaxy?
                               #      True: Orient by even/odd leaf [-pi or +pi]
                               #     False: Orient randomly from    [-pi to +pi]
                 verbose=False,# Print debug outputs
                 Intr=True     # Display in interactive window T/F = Y/N
                 ):
    
        # Save leaf number
        self.LeafN=Morph['Leaf']
        
        # Leaf shape cardinal features
        self.lmax =Morph['lmax']*sqrt(Sx) 
        self.wmax =Morph['wmax']*self.lmax *sqrt(Sx)
        self.swmax=Morph['swmax']
        self.lwmax=self.swmax*self.lmax
        self.wlig =2.*self.wmax/3.
        
        # Save ligule position
        if not Morph['ligule'] is None:
            self.ligule=Morph['ligule'].reshape((3,1))
        else:
            self.ligule=np.zeros((3,1))
        
        # Determine positions and numbers of midrib points
        self.apr_sp=pt_space
        
        num1=max(int(self.lwmax/self.apr_sp)+1,2)
        el1=np.linspace(0.,self.lwmax,num=num1)
        
        num2=max(int((self.lmax-self.lwmax)/self.apr_sp)+1,4)
        el2=np.linspace(self.lwmax,self.lmax,num=num2)

        self.el=np.concatenate((el1,el2[1:]))
        self.es=self.el/self.lmax
        self.dp=self.el.copy(); self.dp[1:]-=self.dp[:-1].copy()
        
        self.NumPts=len(self.el)
        self.Num1=num1
        self.Num2=self.NumPts-self.Num1
        
        # Form factors
        self.f1=Morph['f1']
        self.f2=Morph['f2']
        
        # Leaf width curvature constants
        self.c1=1./self.f1-1.0
        
        r2=lambda c2: self.wlig/self.wmax+(1-self.wlig/self.wmax)*\
                      (1+1/c2-1/log(1+c2))-self.f2
        self.c2=bisect(r2,-0.999,20.,xtol=1e-8)
        
        # Compute leaf widths
        w1=[self.wmax*(s/self.swmax)**self.c1 
                       for s in self.es if s<=self.swmax ]
        w2=[self.wlig+(self.wmax-self.wlig)*log(1.+self.c2*
                       (1-s)/(1-self.swmax))/
                       log(1+self.c2)
                       for s in self.es if s>self.swmax]
        self.w=np.array(w1+w2)        
        
        # 3D midrib launch angles
        self.Th8a1=Morph['Th8a1']
        self.Th8a2=Morph['Th8a2']
        
        # 3D parabolic coefficients
        sec=lambda Theta: 1/cos(Theta)
        self.b1=(tan(self.Th8a1)*sec(self.Th8a1)+\
                 log(tan(self.Th8a1)+sec(self.Th8a1)))/\
                 (4*self.lwmax)
        self.b2=(tan(self.Th8a2)*sec(self.Th8a2)+\
                 log(tan(self.Th8a2)+sec(self.Th8a2)))/\
                 (4*(self.lmax-self.lwmax))
                 
        # 3D parabolic solution functions
        Lb4=lambda T,b,L: T*sqrt(1+T**2)+log(T+sqrt(1+T**2))-4*L*b
        X  =lambda   b,L: bisect(Lb4,0.,1e2,args=(b,L),xtol=1e-8)/(2*b)
        
        # Compute and orient 3D midrib segments
        mrx1= np.array([X(self.b1,L) for L in el1])
        mrx2=-np.array([X(self.b2,L) for L in el2-self.lwmax])
        mrz1=-self.b1*mrx1**2
        mrz2=-self.b2*mrx2**2
        mrx1-=mrx2[-1]; mrz1-=mrz2[-1]
        mrx2-=mrx2[-1]; mrz2-=mrz2[-1]
        
        # Concatenate midrib segments     
        mrx2=reverse(mrx2); mrz2=reverse(mrz2)
        self.mrx=np.concatenate((mrx2,mrx1[1:]))
        self.mrz=np.concatenate((mrz2,mrz1[1:]))
        
        # Reverse lamina edge arrays
        self.el=reverse(self.el)
        self.es=reverse(self.es)
        self.dp=reverse(self.dp)
        
        # Compute areas (taken to be proportional to mass) for
        # each midrib point.  Because ligules do not move their
        # mass is set to zero to maintain indexing correspondence
        # with points.
        
        # Note: An off-by-one error will result in division by
        # zero.  This is not a bad thing because it will make that
        # kind of error easily detectable.
        
        self.Seg_Areas=[0.]; w=reverse(self.w)
        for i,dp in enumerate(self.dp):
            
            # Ligule has area zero (see note above)
            if i==0: continue
            
            # Compute area as two trapezoids with heights that
            # are w/2 and base that is dp
            self.Seg_Areas+=[dp*(w[i-1]+w[i])/2.]
        
        # Compute total leaf area
        self.Seg_Areas=np.array(self.Seg_Areas)
        self.Leaf_Area=np.sum(self.Seg_Areas)
        
        # Make corresponding 3D column vectors
        self.MR=np.zeros((3,self.NumPts))
        self.MR[0,:]=self.mrx
        self.MR[2,:]=self.mrz
        
        # Apply 3D rotation
        r=Morph[ 'roll']            # Roll  between +/-  20 deg
        p=Morph['pitch']            # Pitch +/-  20 deg
        if PhyT:                    # Phyllotaxy requested?
                                    #    Yes - Even/odd leaves => Left/right
            y=0. if even(self.LeafN) else pi
        else: 
            y=uniform(0.,p1)        # yaw between   +/- 180 deg
        
        """ Save this code in case a more refined yaw is needed
        # Apply phyllotaxis to yaw if requested
        if PhyT:
            ya=0.4;yb=1.0               #   ya,yb = semi-minor,semi-major axes
                                        #   i.e across and along rows
            y=atan2(ya*sin(y),yb*cos(y))
        """
        
        R=Rotator(roll=r,pitch=p,yaw=y)    
        self.MR=np.dot(R,self.MR)
        
        # Apply 3D translation
        self.MR+=self.ligule
        
        # Add in a single-leaf vtkPipe
        self.Leaf_Pipe=VTK_Pipe(Intr=Intr)
        
        # debug print
        if verbose:
            print( "Lenth= ",self.lmax)
            print( "Width= ",self.wmax)
            print( "lWmax= ",self.lwmax)
            print( "sWmax= ",self.swmax)
            print( "wlig = ",self.wlig)
            print( "\n")
            print( "Th8a1= ",self.Th8a1*(180/pi),"deg")
            print( "Th8a2= ",self.Th8a2*(180/pi),"deg")
            print( "\n")
            print( "f1    =",self.f1)
            print( "f2    =",self.f2)
            print( "\n")
            print( "c1    =",self.c1)
            print( "c2    =",self.c2)
            print( "ligule=",self.ligule)
            print( "\n")
            print( "Pnts  =",self.NumPts)
        
        # Done
        return
            
    def Move_Leaf(self,Dest):
        # Translate leaf to new position
        
        # Calculate delta
        dest=Dest.reshape((3,1))
        delt=dest-self.ligule
        
        # Move leaf
        self.MR+=delt
        
        # Store new position
        self.ligule=dest
        
        # Done
        return
        
    # Plot leaf in 2D
    def Plot_Leaf_2D(self):
        
        # Make plot
        plt.figure()
        lamina=self.w/2 
        plt.plot(self.el,lamina,'g',self.el,-lamina,'g')
        plt.plot(2*[self.el[-1]],[self.wlig/2,-self.wlig/2],'g')
        plt.axes().set_aspect('equal', 'datalim')
        plt.xlabel("Position along midrib (cm)")
        plt.ylabel("Lamina width (cm)")
        plt.title("Flattened leaf shape")
        plt.show()
        
    # Plot initial midrib profile
    def PLot_Initial_Midrib(self):
        
        # Make plot
        plt.figure()
        plt.plot(self.mrx,self.mrz,'g')
        plt.axes().set_aspect('equal', 'datalim')
        plt.xlabel("x (cm)")
        plt.ylabel("z (cm)")
        plt.title("Initial midrib profile")
        plt.show()
        
    def Plot_Current_Midrib(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(self.MR[0,:],self.MR[1,:],self.MR[2,:])
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")
        ax.set_zlabel("z (cm)")
        plt.show()  

    # Make a vtk leaf
    def VTK_Leaf(self):
        
        """
        The following series of steps leaves a formed leaf as
        the output of the cleanFilter of this leaf's VTK_Pipe
        """
        
        # Step 1 - Get vectors between successive midrib points
        MRDif=self.MR[:,:-1]-self.MR[:,1:]
        MRBeg=np.concatenate((-MRDif[:,+1].reshape((3,1)),MRDif),axis=1)
        MREnd=np.concatenate((MRDif,-MRDif[:,-2].reshape((3,1))),axis=1)
        
        # Step 2 - Compute set of cross products and normalize them
        MRprp=np.cross(MRBeg,MREnd,axis=0)
        MRprp=MRprp/norm(MRprp,axis=0)
        
        # Step 3 - Compute offsets from midrib to an edge
        Offsets=reverse(self.w)*MRprp/2 
        
        # Step 4 - Compute a list of two strips, one for each half leaf
        HalfLeaves=[self.Leaf_Pipe.MakeStrip(self.MR,self.MR+d*Offsets)
                         for d in [-1,1]]
        
        # Step 5 - Combine the two half-leaves and remove duplicate points
        self.Leaf_Pipe.Merge_and_Clean(HalfLeaves)
        
        # Done
        return
    
    # Leaf distructor
    def __del__(self):
        del self.Leaf_Pipe


# Plant object
class Plant(object):
    
    def __init__(self,LeafDicts,Internodes,Intr=True):
        # Link to box for fully grown plant
        global BigBox
        
        """ Error check"""
        if len(LeafDicts)!=len(Internodes):
            raise Exception("There must be as many internodes as leaves")
        
        """ Make a list of leaves """
        # Create leaves
        self.Leaves=[Leaf(di,Intr=Intr) for di in LeafDicts]
        
        """ Add in a plant-level VTK_Pipe """
        self.Plant_Pipe=VTK_Pipe(Intr=Intr)
        
        """ Create a camera and set its clipping range """
        
        self.Plant_Pipe.Camera=vtk.vtkCamera()
        self.Plant_Pipe.Camera.SetViewUp(0.,0.,1.)
        clipping=np.array([-10.,10.])
        self.Plant_Pipe.Camera.SetClippingRange(*(clipping))
        
        """Create set of internodes """
        # Calculate radii
        if len(self.Leaves)>=2:
            Radii=[(self.Leaves[i].wlig+self.Leaves[i+1].wlig)/2
                   for i in range(len(self.Leaves)-1)]
            Radii=[(self.Leaves[0].wlig+Radii[0])/2]+Radii
        else:
            Radii=[self.Leaves[0].wlig]
        Radii=np.array(Radii)/2.
        
        # Calculate Beg/End points
        Pts=[0.]+list(np.cumsum(Internodes)); nP=len(Pts)
        BegEnd=np.zeros((3,nP))
        BegEnd[2,:]=np.array(Pts).reshape((1,nP))

        # Make the internodes
        Color=(0.00,.75,0.00)                   # Green internodes
                                                # Create the set of rods
        self.Plant_Pipe.MakeRodSet(BegEnd[:,0:-1],BegEnd[:,1:],Radii,Color)
        
        # Move the leaves to the ends of the rods
        for i,lf in enumerate(self.Leaves):
            lf.Move_Leaf(BegEnd[:,i+1])
            
        """ Create a set of nodes """
        # Create the Radii
        nE=Radii.shape[0]
        XYZradii=np.zeros((3,nE))
        XYZradii[0,:]=1.1*Radii 
        XYZradii[1,:]=1.0*XYZradii[0,:]
        XYZradii[2,:]=1.2*XYZradii[0,:]
        
        # Specify their centers and desired major axes
        Centers=BegEnd[:,1:]
        MajorAxis=np.array(nE*[[0,0,1]],dtype=float).T
        
        # Create the nodes
        Color=np.array([*Color])  # This is UGLLLEY
        self.Plant_Pipe.MakeEllipsoidSet(Centers,XYZradii,MajorAxis,Color)
        
        """ Position the camera based on the bounding box """
        # If boox for fully grown plant (i.e. last frame which is done first)
        # does not yet exist, then make one
        if BigBox is None: BigBox=self.Bounding_Box()

        Zview=BigBox[2,-1]/2
        Point= lambda th,s: np.array([sin(th)*s,cos(th)*s,Zview])
        cPos=Point(0.8,4*Zview)
        self.Plant_Pipe.Camera.SetPosition(*(cPos))
        self.Plant_Pipe.Camera.SetFocalPoint(0.9*cPos[0],0.9*cPos[1],cPos[2])
        self.Plant_Pipe.renderer.SetActiveCamera(self.Plant_Pipe.Camera)
                
    # Compute bounding box for PLant
    def Bounding_Box(self):

        # Range of x coordinates
        X=np.concatenate([lf.MR[0,:] for lf in self.Leaves])
        x_range=[np.min(X),np.max(X)]
        
        # Range of y coordinates
        Y=np.concatenate([lf.MR[1,:] for lf in self.Leaves])
        y_range=[np.min(Y),np.max(Y)]

        # Range of z coordinates
        Z=np.concatenate([lf.MR[2,:] for lf in self.Leaves])
        z_range=[np.min(Z),np.max(Z)]
        
        # Done
        return np.array([x_range,y_range,z_range])
    
    # Make a set of axes
    def Plot_Axes(self,radius):
        global BigBox
        
        """
        # If the bounding box has not been supplied then get it
        if BigBox is None: BBox=self.Bounding_Box()
        else: BBox=BigBox
        """
        BBox=BigBox
        
        # Construct BegPts, Origin, and EndPts
        BegPts=np.array([[BBox[0,0],       0 ,       0],
                         [       0 ,BBox[1,0],       0],
                         [       0 ,       0 ,BBox[2,0]]])
            
        Origin=np.zeros((3,3))    
        
        EndPts=np.array([[BBox[0,1],       0 ,       0],
                         [       0 ,BBox[1,1],       0],
                         [       0 ,       0 ,BBox[2,1]]])
            
        # Set radii and colors for plus & minus segments
        Radii=[radius,radius,radius]
        pColor=(1.,0.,0.); mColor=(0.,0.,1.)
            
        # Make the negative axes and the positive axes
        self.Plant_Pipe.MakeRodSet(BegPts,Origin,Radii,mColor)
        self.Plant_Pipe.MakeRodSet(Origin,EndPts,Radii,pColor)
        
        # Done - Save the BBox for use in other contexts
        BigBox=BBox
        return
    
    # Delete plant
    def __del__(self):
        for lf in self.Leaves:
            del lf
        del self.Plant_Pipe

    
# Make wheat plant image:
#   Returns Plant object if Intr==True
#           renderwindow object otherwise
def Draw_Plant(LeafDicts,Internodes,Axis=False,Intr=True):

    # Create the plant
    Weeet=Plant(LeafDicts,Internodes,Intr=Intr)
    
    # Render the leaves and connect them to the plant
    for lf in Weeet.Leaves: lf.VTK_Leaf()    
    Weeet.Plant_Pipe.Clean_Some_Cleaners([lf.Leaf_Pipe.cleanFilter 
                                          for lf in Weeet.Leaves])
    # Draw the axes if requested
    if Axis:
        Weeet.Plot_Axes(0.15)
        
    # Finalize the picture and either...
    Picture=Weeet.Plant_Pipe.Fire_for_Effect()
    if Intr: return Weeet    #...display on screen and return plant or...
    else   : 
        del Weeet            #... plant as there will be many more and ...
        return Picture       #...return the image to save in a file

# Main routine
def main():
    
    """ Set up list of dictionaries for two leaves """
    """ In actual use these would come from model"""

    # First emerged leaf
    D1={ 'lmax': 20.,        #  Midrib length [2.,40.]
         'wmax': 0.05,       #  Rel max width position [0.0375,0.0675]
        'swmax': .2,         #  Rel max width [0.1,0.4] 
           'f1': .60,        #  First form factor  [0.55,0.70]
           'f2': .85,        #  Second form factor [0.75,0.90]
        'Th8a1': 5*pi/12,    #  First launch angle  [3pi/12,5pi/12]
        'Th8a2': 5*pi/12,    #  Second launch angle [2pi/12,4pi/12]
         'roll': 0.,         #  Ligule roll angle [-pi/9,+pi/9] 
        'pitch': 0.,         #  Ligule pitch angle [0] (The8a1 governs)
          'yaw': 0.,         #  Ligule yaw angle as determined by PhyT
       'ligule': None,       #  Ligule position 
         'Leaf': 1           #  Leaf number
         }

    # Second emerged leaf
    D2={ 'lmax': 10.,        #  Midrib length [2.,40.]
         'wmax': 0.05,       #  Rel max width position [0.0375,0.0675]
        'swmax': .2,         #  Rel max width [0.1,0.4] 
           'f1': .60,        #  First form factor  [0.55,0.70]
           'f2': .85,        #  Second form factor [0.75,0.90]
        'Th8a1': 3*pi/12,    #  First launch angle  [3pi/12,5pi/12]
        'Th8a2': 4*pi/12,    #  Second launch angle [2pi/12,4pi/12]
         'roll': 0.,         #  Ligule roll angle [-pi/9,+pi/9] 
        'pitch': 0.,         #  Ligule pitch angle [0] (The8a1 governs)
          'yaw': 0.,         #  Ligule yaw angle as determined by PhyT
       'ligule': None,       #  Ligule position 
         'Leaf': 2           #  Leaf number
         }
    LeafDicts=[D1,D2]
    
    """ Set up list of internode lengths """
    """ The first is from ground to first leaf """
    Internodes=[5.,8.]
    
    # Draw the plant & show it on the screen
    Weeet=Draw_Plant(LeafDicts,Internodes,Axis=True,Intr=True)
    
    # Delete the plant so as to free memory
    del Weeet


if __name__=="__main__": main()        
