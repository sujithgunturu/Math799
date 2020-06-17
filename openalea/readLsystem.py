import openalea.lpy as lpy
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import time
from openalea.plantgl.all import *
from openalea.mtg.io import lpy2mtg, mtg2lpy, axialtree2mtg, mtg2axialtree
from openalea.mtg.aml import *
l = lpy.Lsystem('sample1.lpy')
tree = l.iterate() 
l.plot(tree)
#Viewer.frameGL.saveImage('adel2.png', 'png')