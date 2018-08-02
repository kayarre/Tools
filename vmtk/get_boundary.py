
#!/usr/bin/env python

import vtk
import numpy as np
from vmtk import vmtkscripts
import argparse
import copy
import sys

# extract region boundaries
class GetBoundaries():

    def __init__(self, args):

        self.Surface = None
        self.InputFile = args.surface
        self.OutputFile = args.file_out
        self.FeatureAngle = 50.0 #args.feature_angle
        
        self.NumberOfRegions = 0
        
        self.RegionIdsArrayName = "RegionsId"
        self.boundaries = vtk.vtkFeatureEdges() 
        self.NewScalars = vtk.vtkIntArray()
        self.RegionAreas = vtk.vtkDoubleArray()
        self.mesh = vtk.vtkPolyData()
        self.BoundaryLines = vtk.vtkPolyData()
        self.BoundaryPointArray = vtk.vtkIntArray()
        self.BoundaryCellArray = vtk.vtkIntArray()
        self.CheckCells = vtk.vtkIdList()
        self.CheckCells2 = vtk.vtkIdList()
        self.CheckCellsCareful = vtk.vtkIdList()
        self.CheckCellsCareful2 = vtk.vtkIdList()
        
        # dynamic alloated arrays
        self.checked = None
        self.checkedcarefully = None
        self.pointMapper = None
        
        #Feature edges options
        self.BoundaryEdges  = 0
        self.ManifoldEdges = 0
        self.NonManifoldEdges = 0
        self.FeatureEdges = 1

        self.ExtractLargestregion = 0
        
    """ brief Function to flood fill region fast. """
    def FindBoundaryRegion(self, reg, start):
        
        #Variables used in function
        # i 
        # j,k,l,cellId
        #vtkIdType *pts = 0
        #vtkIdType npts = 0
        #vtkIdType numNei, nei, p1, p2, nIds, neis
        npts = 0

        #Id List to store neighbor cells for each set of nodes and a cell
        neighbors = vtk.vtkIdList()
        tmp = vtk.vtkIdList()
        #pts = vtk.vtkIdList()
        #Variable for accessing neiIds list
        sz = 0 #vtkIdType

        #Variables for the boundary cells adjacent to the boundary point
        bLinesOne = vtk.vtkIdList()
        bLinesTwo = vtk.vtkIdList()

        numCheckCells = 0
        pts = vtk.vtkIdList()
        # Get neighboring cell for each pair of points in current cell
        while (self.CheckCells.GetNumberOfIds() > 0):
            numCheckCells = self.CheckCells.GetNumberOfIds()
            print("peace", numCheckCells)
            for c in range(numCheckCells):
                cellId = self.CheckCells.GetId(c)
                #Get the three points of the cell
                self.mesh.GetCellPoints(cellId,pts)
                
                if (self.checked.GetValue(cellId) == 0):
                    #Mark cell as checked and insert the fillnumber value to cell
                    self.NewScalars.InsertValue(cellId,reg)
                    self.checked.SetValue(cellId, 1)
                    for i in range(pts.GetNumberOfIds()):
                        p1 = pts.GetId(i)
                        #Get the cells attached to each point
                        self.mesh.GetPointCells(p1,neighbors)
                        numNei = neighbors.GetNumberOfIds()
                        #print(numNei)
                        #For each neighboring cell
                        for j in range(numNei):
                            #print(self.BoundaryCellArray.GetValue(neighbors.GetId(j)))
                            #If this cell is close to a boundary
                            if (self.BoundaryCellArray.GetValue(neighbors.GetId(j))):
                                #If self cell hasn't been checked already
                                if (self.checkedcarefully.GetValue(neighbors.GetId(j)) == 0):
                                    print("test hoop")
                                    #Add self cell to the careful check cells list and run
                                    #the region finding tip toe code
                                    self.CheckCellsCareful.InsertNextId(neighbors.GetId(j))
                                    self.FindBoundaryRegionTipToe(reg)

                                    self.CheckCellsCareful.Reset()
                                    self.CheckCellsCareful2.Reset()

                            #Cell needs to be added to check list
                            else:
                                self.CheckCells2.InsertNextId(neighbors.GetId(j))
                
                #If the start cell is a boundary cell
                elif (self.checkedcarefully.GetValue(cellId) == 0 and start):
                    #Reset the check cell list and start a careful search
                    start=0
                    self.CheckCells.Reset()
                    print("I have been added begin {0}".format(cellId))
                    self.CheckCellsCareful.InsertNextId(cellId)
                    self.FindBoundaryRegionTipToe(reg)
                
                 
            
            #Swap the current check list to the full check list and continue
            tmp.DeepCopy(self.CheckCells)
            self.CheckCells.DeepCopy( self.CheckCells2)
            self.CheckCells2.DeepCopy(tmp)
            tmp.Reset()

        
    """   Function to flood fill region slower, but is necessary close 
    to boundaries to make sure it doesn't step over boundary.
    """
    def FindBoundaryRegionTipToe(self, reg):
        #Variables used in function
        #int i 
        #vtkIdType j,k,l 
        #vtkIdType *pts = 0 
        #vtkIdType npts = 0 
        #vtkIdType cellId 
        #vtkIdType numNei, nei, p1, p2, nIds, neiId 

        #Id List to store neighbor cells for each set of nodes and a cell
        tmp = vtk.vtkIdList()
        neiIds = vtk.vtkIdList()

        #Variable for accessing neiIds list
        sz = 0 

        #Variables for the boundary cells adjacent to the boundary point
        bLinesOne = vtk.vtkIdList()
        bLinesTwo = vtk.vtkIdList()

        numCheckCells = 0.0
        pts = vtk.vtkIdList()
        #Get neighboring cell for each pair of points in current cell
        #While there are still cells to be checked
        while ( self.CheckCellsCareful.GetNumberOfIds() > 0):
            numCheckCells =  self.CheckCellsCareful.GetNumberOfIds()
            for  c in range(numCheckCells):
                neiIds.Reset() 
                cellId =  self.CheckCellsCareful.GetId(c) 
                #Get the three points of the cell
                self.mesh.GetCellPoints(cellId,pts) 
                if ( self.checkedcarefully.GetValue(cellId) == 0):
                
                    #Update this cell to have been checked carefully and assign it
                    #with the fillnumber scalar
                    self.NewScalars.InsertValue(cellId,reg) 
                    self.checkedcarefully.SetValue(cellId, 1) 
                    #For each edge of the cell
                    print("Checking edges of cell {0}".format(cellId))
                    for i in range(pts.GetNumberOfIds()):
                    
                        p1 = pts.GetId(i) 
                        p2 = pts.GetId((i+1)%(pts.GetNumberOfIds())) 

                        neighbors = vtk.vtkIdList()
                        
                        #Initial check to make sure the cell is in fact a face cell
                        self.mesh.GetCellEdgeNeighbors(cellId,p1,p2,neighbors) 
                        numNei = neighbors.GetNumberOfIds() 

                        #Check to make sure it is an oustide surface cell,
                        #i.e. one neighbor
                        if (numNei==1):
                            count = 0 
                            #Check to see if cell is on the boundary,
                            #if it is get adjacent lines
                            if ( self.BoundaryPointArray.GetValue(p1) == 1):
                                count += 1 

                            if ( self.BoundaryPointArray.GetValue(p2) == 1):
                                count += 1 

                            nei=neighbors.GetId(0) 
                            #if cell is not on the boundary, add new cell to check list
                            if (count < 2):
                                neiIds.InsertNextId(nei) 

                            #if cell is on boundary, check to make sure it isn't
                            #false positive  don't add to check list.  self is done by
                            #getting the boundary lines attached to each point, then
                            #intersecting the two lists. If the result is zero, then  self
                            #is a false positive
                            else:
                                self.BoundaryLines.BuildLinks() 
                                bPt1 = self.pointMapper.GetPoint(p1) 
                                self.BoundaryLines.GetPointCells(bPt1,bLinesOne) 

                                bPt2 = self.pointMapper.GetPoint(p2) 
                                self.BoundaryLines.GetPointCells(bPt2,bLinesTwo) 

                                bLinesOne.IntersectWith(bLinesTwo) 
                                #Cell is false positive. Add to check list.
                                if (bLinesOne.GetNumberOfIds() == 0):
                                    neiIds.InsertNextId(nei) 
                                                
                    nIds = neiIds.GetNumberOfIds() 
                    if (nIds>0):
                        #Add all Ids in current list to global list of Ids
                        for k in range(nIds):
                            neiId = neiIds.GetId(k) 
                            if ( self.checkedcarefully.GetValue(neiId)==0):
                                self.CheckCellsCareful2.InsertNextId(neiId)
                            elif ( self.checked.GetValue(neiId)==0):
                                self.CheckCells2.InsertNextId(neiId) 
                            
            #Add current list of checked cells to the full list and continue
            tmp.DeepCopy(self.CheckCells)
            self.CheckCells.DeepCopy( self.CheckCells2)
            self.CheckCells2.DeepCopy(tmp)
            tmp.Reset()
            #print("here") 
        
    """ Initializes boundary arrays. """
    def SetBoundaryArrays(self):
        #Variables used in the function
        #vtkIdType pointId,bp,bp2,i
        bpCellIds = vtk.vtkIdList()
        
        #Point locator to find points on mesh that are the points on the boundary
        #lines
        pointLocator = vtk.vtkPointLocator()
        pointLocator.SetDataSet(self.mesh)
        pointLocator.BuildLocator()

        # Get number of points and cells
        numMeshPoints = self.mesh.GetNumberOfPoints()
        numMeshCells = self.mesh.GetNumberOfCells()

        # Set up check arrays
        self.checked = vtk.vtkIdTypeArray()
        self.checked.SetNumberOfComponents(1)
        self.checked.SetNumberOfTuples(numMeshCells)# vtkIdType[numMeshCells]
        self.checked.Fill(0.0)
        
        self.checkedcarefully = vtk.vtkIdTypeArray()
        self.checkedcarefully.SetNumberOfComponents(1)
        self.checkedcarefully.SetNumberOfTuples(numMeshCells)# vtkIdType[numMeshCells]
        self.checkedcarefully.Fill(1.0)
        
        self.pointMapper = vtk.vtkIdTypeArray()
        self.pointMapper.SetNumberOfComponents(1)
        self.pointMapper.SetNumberOfTuples(numMeshCells)# vtkIdType[numMeshCells]
        self.pointMapper.Fill(0.0)

        #for i in range(numMeshCells):
        #    self.checked.SetValue(i, 0.)

        # Set up boundary arrays
        self.BoundaryPointArray.SetNumberOfComponents(1)
        self.BoundaryPointArray.SetNumberOfTuples(numMeshPoints)
        self.BoundaryPointArray.FillComponent(0, 0)
        self.BoundaryCellArray.SetNumberOfComponents(1)
        self.BoundaryCellArray.SetNumberOfTuples(numMeshCells)
        self.BoundaryCellArray.FillComponent(0,0)

        # Number of boundary line points
        numPoints = self.BoundaryLines.GetNumberOfPoints()
        pt = [0.0, 0.0, 0.0]
        for pointId in range(numPoints):
            self.BoundaryLines.GetPoint(pointId, pt)
            print(pt)
            #Find point on mesh
            bp = pointLocator.FindClosestPoint(pt)
            self.pointMapper.SetValue(bp, pointId)
            self.BoundaryPointArray.SetValue(bp, 1)
            self.mesh.GetPointCells(bp,bpCellIds)
            #Set the point mapping array
            #Assign each cell attached to self point as a boundary cell
            for  i in range(bpCellIds.GetNumberOfIds()):
                self.BoundaryCellArray.SetValue(bpCellIds.GetId(i), 1)
                #print(self.BoundaryCellArray.GetValue(bpCellIds.GetId(i)))
                self.checked.InsertValue(bpCellIds.GetId(i), 1)

        # Flip the values of checked carefully
        for  i in range(numMeshCells):
            if (self.checked.GetValue(i) == 0):
                self.checkedcarefully.SetValue(i, 1)
            else:
                self.checkedcarefully.SetValue(i, 0)
        
    """ function to add current cell area to full area.
        cellId cell whose are to be computed.
        area area which will be updated with cell area.
   """
    def AddCellArea(self, cellId, area):

        # Get cell points
        #vtkIdType npts, *pts
        pts = vtk.vtkIdList() 
        self.mesh.GetCellPoints(cellId, pts) 

        # Get points
        pt0 = [0.0, 0.0, 0.0]
        pt1 = [0.0, 0.0, 0.0]
        pt2 = [0.0, 0.0, 0.0]
        self.mesh.GetPoint(pts.GetId(0), pt0) 
        self.mesh.GetPoint(pts.GetId(1), pt1) 
        self.mesh.GetPoint(pts.GetId(2), pt2) 

        # Calculate area of triangle
        area += abs(self.ComputeTriangleArea(pt0, pt1, pt2))
        
    """
    ComputeTriangleArea
    """ 
    def ComputeTriangleArea(self, pt0, pt1, pt2):
        area = 0.0
        area += (pt0[0]*pt1[1])-(pt1[0]*pt0[1])
        area += (pt1[0]*pt2[1])-(pt2[0]*pt1[1])
        area += (pt2[0]*pt0[1])-(pt0[0]*pt2[1])
        area *= 0.5
        return area

    def Execute(self):
        print("Get Surface Boundaries")
    
        reader = vmtkscripts.vmtkSurfaceReader()
        reader.InputFileName = self.InputFile
        reader.Execute()
        
        self.Surface = reader.Surface #vtkPolyData
                
        #Define variables used by the algorithm
        inpts= vtk.vtkPoints()
        inPolys = vtk.vtkCellArray()
        
        # ints of vtkIdType
        # newPts  numPts, newId, cellId
        
        #Get input points, polys and set the up in the vtkPolyData mesh
        inpts = self.Surface.GetPoints()
        inPolys = self.Surface.GetPolys()
        
        self.mesh.SetPoints(inpts)
        self.mesh.SetPolys(inPolys)
        
        #Build Links in the mesh to be able to perform complex polydata processes
        
        self.mesh.BuildLinks()
        
        #Get the number of Polys for scalar  allocation
        numPolys = self.Surface.GetNumberOfPolys()
        numPts = self.Surface.GetNumberOfPoints()
        
        #Check the input to make sure it is there
        if (numPolys < 1):
            raise RuntimeError("No Input")



        #Set up Region scalar for each surface
        self.NewScalars.SetNumberOfTuples(numPolys)

        #Set up Feature Edges for Boundary Edge Detection
        inputCopy = self.Surface.NewInstance()
        inputCopy.ShallowCopy(self.Surface)

        #Set the Data to hold onto given Point Markers
        inputCopy.GlobalReleaseDataFlagOff()
        self.boundaries.SetInputData(inputCopy)
        self.boundaries.BoundaryEdgesOff() #(self.BoundaryEdges)
        self.boundaries.ManifoldEdgesOff() #(self.ManifoldEdges)
        self.boundaries.NonManifoldEdgesOff() #(self.NonManifoldEdges)
        self.boundaries.FeatureEdgesOn() #(self.FeatureEdges)
        self.boundaries.SetFeatureAngle(self.FeatureAngle)
        #inputCopy.Delete()
        self.boundaries.Update()

        # Set the boundary lines
        self.BoundaryLines.DeepCopy(self.boundaries.GetOutput())

        # Initialize the arrays to be used in the flood fills
        self.SetBoundaryArrays()

        print("Starting Boundary Face Separation")
        # Set Region value of each cell to be zero initially
        reg = 0
        for cellId in range(numPolys):
            self.NewScalars.InsertValue(cellId, reg)


        #Go through each cell and perfrom region identification proces
        #print(numPolys)
        for cellId in range(numPolys):
            #if(cellId % 1000 == 0):
                #print(cellId)
            #Check to make sure the value of the region at self cellId hasn't been set
            if (self.NewScalars.GetValue(cellId) == 0):
                reg += 1
                self.CheckCells.InsertNextId(cellId)
                #Call function to find all cells within certain region
                self.FindBoundaryRegion(reg, 1)
                #print("party")
                self.CheckCells.Reset()
                self.CheckCells2.Reset()
                self.CheckCellsCareful.Reset()
                self.CheckCellsCareful2.Reset()

        # Check to see if anything left
        extraregion=0
        for cellId in range(numPolys):
            if (self.checked.GetValue(cellId) == 0 or self.checkedcarefully.GetValue(cellId == 0)):
                self.NewScalars.InsertValue(cellId,reg+1)
                self.AddCellArea(cellId, area)
                extraregion=1

        if (extraregion):
            print("I am incrementing region")
            reg +=1

        #Copy all the input geometry and data to the output
        output.SetPoints(inpts)
        output.SetPolys(inPolys)
        output.GetPointData().PassData(input.GetPointData())
        output.GetCellData().PassData(input.GetCellData())

        #Add the new scalars array to the output
        self.NewScalars.SetName(self.RegionIdsArrayName)
        output.GetCellData().AddArray(self.NewScalars)
        output.GetCellData().SetActiveScalars(self.RegionIdsArrayName)

        # If extracting largets region, get it out
        if (self.ExtractLargestRegion):
            maxVal = 0.0
            maxRegion = -1
            for i in range(reg):
                if (self.RegionAreas.GetValue(i) > maxVal):
                    maxVal = self.RegionAreas.GetValue(i)
                    maxRegion = i+1
            
            thresholder = vtk.vtkThreshold()
            thresholder.SetIntputData(output)
            thresholder.SetInputArrayToProcess(0, 0, 0, 1, self.RegionIdsArrayName)
            thresholder.ThresholdBetween(maxRegion, maxRegion)
            thresholder.Update()
            
            # Check to see if the result has points, don't run surface filter
            if (thresholder.GetOutput().GetNumberOfPoints() == 0):
                raise RuntimeError("vtkThreshold Output has no points")
            
            #Convert unstructured grid to polydata
            surfacer = vtk.vtkDataSetSurfaceFilter()
            surfacer.SetInputData(thresholder.GetOutput())
            surfacer.Update()
            
            #Set the final pd
            output.DeepCopy(surfacer.GetOutput())
            
        # Total number of regions
        self.NumberOfRegions = reg   

        writer = vmtkscripts.vmtkSurfaceWriter()
        writer.OutputFileName = self.OutputFile
        writer.Input = output
        writer.Execute()





if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Get surface boundaries')
    parser.add_argument("-i", dest="surface", required=True, help="input surface file", metavar="FILE")
    parser.add_argument("-o", dest="file_out", required=True, help="output surface file", metavar="FILE")
    #parser.add_argument("-a", dest="feature_angle", required=False, help="feature_angle", metavar="FILE", default=50.0)

    args = parser.parse_args()
    #print(args)
    boundaries = GetBoundaries(args)
    boundaries.Execute()


