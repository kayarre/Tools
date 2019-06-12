import vtk
from vmtk import vtkvmtk

def ReadPolyData(filename):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

def WritePolyData(input,filename):
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(input)
    writer.Write()

polyBallImageSize = [256,256, 256]
radiusArrayName = 'MaximumInscribedSphereRadius'
parallelTransportNormalsArrayName = 'ParallelTransportNormals'
surfaceFilename = "test_reconstruction.vtp"

completeVoronoiDiagram = ReadPolyData("test_2.vtp")

print 'Reconstructing Surface from Voronoi Diagram'
modeller = vtkvmtk.vtkvmtkPolyBallModeller()
modeller.SetInputData(completeVoronoiDiagram)
modeller.SetRadiusArrayName(radiusArrayName)
modeller.UsePolyBallLineOff()
modeller.SetSampleDimensions(polyBallImageSize)
modeller.Update()

marchingCube = vtk.vtkMarchingCubes()
marchingCube.SetInputData(modeller.GetOutput())
marchingCube.SetValue(0,0.0)
marchingCube.Update()
envelope = marchingCube.GetOutput()
WritePolyData(envelope,surfaceFilename)
