#!/usr/bin/env python

## Program:   KVMTK
## Module:    $RCSfile: vmtksurfacearea.py,v $
## Language:  Python
## Date:      $Date: 2018/04/19 08:27:40 $
## Version:   $Revision: 1.4 $

##   Copyright (c) Kurt Sansom. All rights reserved.
##   See LICENCE file for details.

##      This software is distributed WITHOUT ANY WARRANTY; without even 
##      the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
##      PURPOSE.  See the above copyright notices for more information.


from __future__ import absolute_import #NEEDS TO STAY AS TOP LEVEL MODULE FOR Py2-3 COMPATIBILITY
import vtk
import numpy as np
#from vmtk import vtkvmtk
import sys

from vmtk import pypes


class vmtkSurfaceArea(pypes.pypeScript):

    def __init__(self):

        pypes.pypeScript.__init__(self)

        self.Surface = None
        self.OutputFileName = ''
        self.EdgeLength = 0.08 #mm in this case
        
        self.SetScriptName('vmtksurfacearea')
        self.SetScriptDoc('estimate the number points for a given triangle size')
        self.SetInputMembers([
            ['Surface','i','vtkPolyData',1,'','the input surface','vmtksurfacereader'],
            ['EdgeLength','value','float',1,'','target edge length to compare against'],
            ['OutputFileName','ofile','str',1,'','output file name']
            ])
        self.SetOutputMembers([])
        #self.SetOutputMembers([
        #    ['Surface','o','vtkPolyData',1,'','the output surface','vmtksurfacewriter']
        #    ])

   
 
    def Execute(self):

        if self.Surface == None:
            if self.Input == None:
                self.PrintError('Error: no Surface.')
            self.Surface = self.Input

        # estimates surface area to estimate the point density

        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(self.Surface)
        cleaner.Update()

        triangleFilter = vtk.vtkTriangleFilter()
        triangleFilter.SetInputConnection(cleaner.GetOutputPort())
        triangleFilter.Update()

        massProps = vtk.vtkMassProperties()
        massProps.SetInputConnection(triangleFilter.GetOutputPort())
        massProps.Update()

        print(massProps.GetSurfaceArea())

        area = massProps.GetSurfaceArea()

        target_area = 3.0**0.5/4.0*self.EdgeLength**2.0

        print ("target number of cells: {0}".format(area / target_area)) # A_total = N*(area_equilateral_triangle)

        print ("target number of points: {0}".format(area / target_area / 2.0)) #in the limit of equilateral triangles ratio ,Ncells/Npoints = 2


if __name__=='__main__':
    main = pypes.pypeMain()
    print(main, sys.argv)
    main.Arguments = sys.argv
    main.Execute()
