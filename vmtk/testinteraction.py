
#!/usr/bin/env python

import vtk
import numpy as np
from vmtk import vmtkscripts
import argparse
import copy

# create an interactor test on a surface
def pick_points(args):
    print("pick stuff")
    
    reader = vmtkscripts.vmtkSurfaceReader()
    reader.InputFileName = args.surface
    reader.Execute()
    Surface = reader.Surface    

    #Program:   Visualization Toolkit
    #Module:    TestGenericCell.cxx
    # This test demonstrates the vtkCellCentersPointPlacer. The placer may
    #be used to constrain handle widgets to the centers of cells. Thus it
    # may be used by any of the widgets that use the handles (distance, angle
    # etc).
    # Here we demonstrate constraining the distance widget to the centers   
    # of various cells.
    
    surfaceactor = vtk.vtkActor()
    #Visualize
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputData(Surface)

    surfaceactor.SetMapper(mapper)
    

    gridDimensions = int(1)
    rendererSize = int(600)

    # Create a render window, renderer and render window interactor.
    # Add the cells to the renderer, in a grid layout. We accomplish    
    # this by using a transform filter to translate and arrange on
    # a grid.

    renderer = vtk.vtkRenderer()
    renderWindow =vtk.vtkRenderWindow()
    renderWindow.SetSize( rendererSize*gridDimensions, rendererSize*gridDimensions)
    renderWindow.AddRenderer(renderer)
    renderer.SetBackground(.2, .3, .4)
    
    #Create a point placer to constrain to the cell centers and add
    # each of the actors to the placer, so that it includes them in
    #its constraints.
    pointPlacer = vtk.vtkCellCentersPointPlacer()
    

    renderer.AddActor(surfaceactor)
    pointPlacer.AddProp(surfaceactor)
    
    #Default colors
    surfaceactor.GetProperty().SetColor(1,0,0.5)

    renderer.ResetCamera()
    renderer.GetActiveCamera().Azimuth(30)
    renderer.GetActiveCamera().Elevation(-30)
    renderer.ResetCameraClippingRange()
    
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
    style = vtk.vtkInteractorStyleTrackballCamera()
    renderWindowInteractor.SetInteractorStyle(style)
    renderWindow.Render()
    
    #Now add a distance widget.

    seedwidget = vtk.vtkSeedWidget()phereWidget2()
    
    #print(dir(widget))
    #widget.CreateDefaultRepresentation()
    rep = vtk.vtkSphereRepresentation()
    print(dir(rep))
    #rep.GetAxis().GetProperty().SetColor( 1.0, 0.0, 0.0 )
    
    # Create a 3D handle representation template for this distance
    # widget
    handleRep3D = vtk.vtkSphereHandleRepresentation()
    handleRep3D.GetProperty().SetColor( 0.8, 0.2, 0 )
    #print(dir(handleRep3D))
    #vtkSmartPointer< vtkPointHandleRepresentation3D >::New();
    #handleRep3D.GetProperty().SetLineWidth(4.0)
    #rep.SetHandleRepresentation( handleRep3D )
    #handleRep3D.AllOn()
    #handleRep3D.GetProperty().SetColor( 0.8, 0.2, 0 )
    #rep.GetProperty().SetColor( 0.8, 0.2, 0 )
   #rep.GetProperty().SetHandleRepresentation(handleRep3D)

    widget.CreateDefaultRepresentation()
    widget.SetRepresentation(rep)
    widget.SetInteractor(renderWindowInteractor)
    #widget.On()
    #widget.ProcessEventsOff()
    
    # Instantiate the handles and have them be constrained by the placer.
    #rep.InstantiateHandleRepresentation()
    print(dir(rep))
    rep.GetHandleRepresentation().SetPointPlacer(pointPlacer)
    #rep.GetPoint2Representation().SetPointPlacer(pointPlacer)
    print(dir(rep.GetHandleRepresentation()))
    # With a "snap" constraint, we can't have a smooth motion anymore, so turn it off.
    #rep.GetHandleRepresentation().SmoothMotionOff()
    
    #vtkPointHandleRepresentation3D
    #rep.GetPoint2Representation().SmoothMotionOff()

    widget.SetInteractor(renderWindowInteractor)
    widget.SetEnabled(1)

    renderWindow.Render()
    
    renderWindowInteractor.Start()


    #writer = vmtkscripts.vmtkSurfaceWriter()
    #writer.OutputFileName = args.file_out
    #writer.Input = warp.GetOutput()
    #writer.Execute()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='estimate vertices  for uniform point distribution')
    parser.add_argument("-i", dest="surface", required=True, help="input surface file", metavar="FILE")
    #parser.add_argument("-o", dest="file_out", required=True, help="output surface file", metavar="FILE")
    #parser.add_argument("-s", '--scale', dest="fuzz_scale",  type=float, help='how much to fuzz surface ', default=0.08)
    args = parser.parse_args()
    #print(args)
    pick_points(args)
