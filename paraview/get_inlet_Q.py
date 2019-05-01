# trace generated using paraview version 5.5.0

#import the simple module from the paraview
from paraview.simple import *
import vtk
# disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

case_list = ["case1", "case3", "case4", "case5", "case7", "case8", "case12", "case13", "case14"]
for case in case_list:

    # create a new 'XML Unstructured Grid Reader'
    inlet_outfile_nodevtu = XMLUnstructuredGridReader(FileName=['/raid/sansomk/caseFiles/mri/VWI_proj/{0}/fluent_dsa/vtk_out/inlet_outfile_node.vtu'.format(case)])
    inlet_outfile_nodevtu.PointArrayStatus = ['velocity']
    #inlet_outfile_nodevtu.PointArrayStatus = ['absolute_pressure', 'velocity', 'x_velocity', 'x_wall_shear', 'y_velocity', 'y_wall_shear', 'z_velocity', 'z_wall_shear']

    # create a new 'Extract Surface'
    extractSurface2 = ExtractSurface(Input=inlet_outfile_nodevtu)

    # create a new 'Generate Surface Normals'
    generateSurfaceNormals2 = GenerateSurfaceNormals(Input=extractSurface2)

    # create a new 'Calculator'
    calculator3 = Calculator(Input=generateSurfaceNormals2)
    calculator3.ResultArrayName = 'vdotn'
    calculator3.Function = 'velocity_X*-Normals_X+velocity_Y*-Normals_Y+velocity_Z*-Normals_Z'


    # create a new 'Integrate Variables'
    integrateVariables2 = IntegrateVariables(Input=calculator3)

    # should be area
    area =  integrateVariables2.CellData.GetArray(0).GetRange()[0]

    diameter = 2.0*(area/vtk.vtkMath.Pi())**0.5

    # create a new 'Calculator'
    calculator4 = Calculator(Input=integrateVariables2)
    # Properties modified on calculator4
    calculator4.ResultArrayName = 'Q'#.format(case)
    calculator4.Function = 'vdotn*10^6*60'


    calculator5 = Calculator(Input=calculator4)
    calculator5.ResultArrayName = 'Re'# .format(case)
    calculator5.Function = "vdotn/{0}*1050.0/0.0035*{1}".format(area, diameter)



    # create a new 'Temporal Shift Scale'
    temporalShiftScale1 = TemporalShiftScale(Input=calculator5)


    # Properties modified on temporalShiftScale1
    temporalShiftScale1.Scale = 0.012

    # create a new 'Plot Data Over Time'
    plotDataOverTime2 = PlotDataOverTime(Input=temporalShiftScale1)
    # Properties modified on plotDataOverTime2
    plotDataOverTime2.OnlyReportSelectionStatistics = 0



    # find view
    renderView1 = FindViewOrCreate('RenderView1', viewtype='RenderView')
    # uncomment following to set a specific view size
    # renderView1.ViewSize = [975, 1165]

    # set active view
    SetActiveView(renderView1)


    # set active source
    SetActiveSource(plotDataOverTime2)


    # find view
    quartileChartView1 = FindViewOrCreate('QuartileChartView1', viewtype='QuartileChartView')
    # uncomment following to set a specific view size
    # quartileChartView1.ViewSize = [974, 567]

    # set active view
    SetActiveView(quartileChartView1)


    # show data in view
    plotDataOverTime2Display = Show(plotDataOverTime2, quartileChartView1)
    

    ## trace defaults for the display properties.
    #plotDataOverTime2Display.AttributeType = 'Row Data'
    #plotDataOverTime2Display.UseIndexForXAxis = 0
    #plotDataOverTime2Display.XArrayName = 'Time'
    #plotDataOverTime2Display.SeriesVisibility = ['absolute_pressure (stats)', 'Normals (Magnitude) (stats)', 'Q (stats)', 'vdotn (stats)', 'velocity (Magnitude) (stats)', 'X (stats)', 'x_velocity (stats)', 'x_wall_shear (stats)', 'Y (stats)', 'y_velocity (stats)', 'y_wall_shear (stats)', 'Z (stats)', 'z_velocity (stats)', 'z_wall_shear (stats)']
    #plotDataOverTime2Display.SeriesLabel = ['absolute_pressure (stats)', 'absolute_pressure (stats)', 'Normals (0) (stats)', 'Normals (0) (stats)', 'Normals (1) (stats)', 'Normals (1) (stats)', 'Normals (2) (stats)', 'Normals (2) (stats)', 'Normals (Magnitude) (stats)', 'Normals (Magnitude) (stats)', 'Q (stats)', 'Q (stats)', 'vdotn (stats)', 'vdotn (stats)', 'velocity (0) (stats)', 'velocity (0) (stats)', 'velocity (1) (stats)', 'velocity (1) (stats)', 'velocity (2) (stats)', 'velocity (2) (stats)', 'velocity (Magnitude) (stats)', 'velocity (Magnitude) (stats)', 'X (stats)', 'X (stats)', 'x_velocity (stats)', 'x_velocity (stats)', 'x_wall_shear (stats)', 'x_wall_shear (stats)', 'Y (stats)', 'Y (stats)', 'y_velocity (stats)', 'y_velocity (stats)', 'y_wall_shear (stats)', 'y_wall_shear (stats)', 'Z (stats)', 'Z (stats)', 'z_velocity (stats)', 'z_velocity (stats)', 'z_wall_shear (stats)', 'z_wall_shear (stats)', 'N (stats)', 'N (stats)', 'Time (stats)', 'Time (stats)', 'vtkValidPointMask (stats)', 'vtkValidPointMask (stats)']
    #plotDataOverTime2Display.SeriesColor = ['absolute_pressure (stats)', '0', '0', '0', 'Normals (0) (stats)', '0.89', '0.1', '0.11', 'Normals (1) (stats)', '0.22', '0.49', '0.72', 'Normals (2) (stats)', '0.3', '0.69', '0.29', 'Normals (Magnitude) (stats)', '0.6', '0.31', '0.64', 'Q (stats)', '1', '0.5', '0', 'vdotn (stats)', '0.65', '0.34', '0.16', 'velocity (0) (stats)', '0', '0', '0', 'velocity (1) (stats)', '0.89', '0.1', '0.11', 'velocity (2) (stats)', '0.22', '0.49', '0.72', 'velocity (Magnitude) (stats)', '0.3', '0.69', '0.29', 'X (stats)', '0.6', '0.31', '0.64', 'x_velocity (stats)', '1', '0.5', '0', 'x_wall_shear (stats)', '0.65', '0.34', '0.16', 'Y (stats)', '0', '0', '0', 'y_velocity (stats)', '0.89', '0.1', '0.11', 'y_wall_shear (stats)', '0.22', '0.49', '0.72', 'Z (stats)', '0.3', '0.69', '0.29', 'z_velocity (stats)', '0.6', '0.31', '0.64', 'z_wall_shear (stats)', '1', '0.5', '0', 'N (stats)', '0.65', '0.34', '0.16', 'Time (stats)', '0', '0', '0', 'vtkValidPointMask (stats)', '0.89', '0.1', '0.11']
    #plotDataOverTime2Display.SeriesPlotCorner = ['absolute_pressure (stats)', '0', 'Normals (0) (stats)', '0', 'Normals (1) (stats)', '0', 'Normals (2) (stats)', '0', 'Normals (Magnitude) (stats)', '0', 'Q (stats)', '0', 'vdotn (stats)', '0', 'velocity (0) (stats)', '0', 'velocity (1) (stats)', '0', 'velocity (2) (stats)', '0', 'velocity (Magnitude) (stats)', '0', 'X (stats)', '0', 'x_velocity (stats)', '0', 'x_wall_shear (stats)', '0', 'Y (stats)', '0', 'y_velocity (stats)', '0', 'y_wall_shear (stats)', '0', 'Z (stats)', '0', 'z_velocity (stats)', '0', 'z_wall_shear (stats)', '0', 'N (stats)', '0', 'Time (stats)', '0', 'vtkValidPointMask (stats)', '0']
    #plotDataOverTime2Display.SeriesLabelPrefix = ''
    #plotDataOverTime2Display.SeriesLineStyle = ['absolute_pressure (stats)', '1', 'Normals (0) (stats)', '1', 'Normals (1) (stats)', '1', 'Normals (2) (stats)', '1', 'Normals (Magnitude) (stats)', '1', 'Q (stats)', '1', 'vdotn (stats)', '1', 'velocity (0) (stats)', '1', 'velocity (1) (stats)', '1', 'velocity (2) (stats)', '1', 'velocity (Magnitude) (stats)', '1', 'X (stats)', '1', 'x_velocity (stats)', '1', 'x_wall_shear (stats)', '1', 'Y (stats)', '1', 'y_velocity (stats)', '1', 'y_wall_shear (stats)', '1', 'Z (stats)', '1', 'z_velocity (stats)', '1', 'z_wall_shear (stats)', '1', 'N (stats)', '1', 'Time (stats)', '1', 'vtkValidPointMask (stats)', '1']
    #plotDataOverTime2Display.SeriesLineThickness = ['absolute_pressure (stats)', '2', 'Normals (0) (stats)', '2', 'Normals (1) (stats)', '2', 'Normals (2) (stats)', '2', 'Normals (Magnitude) (stats)', '2', 'Q (stats)', '2', 'vdotn (stats)', '2', 'velocity (0) (stats)', '2', 'velocity (1) (stats)', '2', 'velocity (2) (stats)', '2', 'velocity (Magnitude) (stats)', '2', 'X (stats)', '2', 'x_velocity (stats)', '2', 'x_wall_shear (stats)', '2', 'Y (stats)', '2', 'y_velocity (stats)', '2', 'y_wall_shear (stats)', '2', 'Z (stats)', '2', 'z_velocity (stats)', '2', 'z_wall_shear (stats)', '2', 'N (stats)', '2', 'Time (stats)', '2', 'vtkValidPointMask (stats)', '2']
    #plotDataOverTime2Display.SeriesMarkerStyle = ['absolute_pressure (stats)', '0', 'Normals (0) (stats)', '0', 'Normals (1) (stats)', '0', 'Normals (2) (stats)', '0', 'Normals (Magnitude) (stats)', '0', 'Q (stats)', '0', 'vdotn (stats)', '0', 'velocity (0) (stats)', '0', 'velocity (1) (stats)', '0', 'velocity (2) (stats)', '0', 'velocity (Magnitude) (stats)', '0', 'X (stats)', '0', 'x_velocity (stats)', '0', 'x_wall_shear (stats)', '0', 'Y (stats)', '0', 'y_velocity (stats)', '0', 'y_wall_shear (stats)', '0', 'Z (stats)', '0', 'z_velocity (stats)', '0', 'z_wall_shear (stats)', '0', 'N (stats)', '0', 'Time (stats)', '0', 'vtkValidPointMask (stats)', '0']

    # Properties modified on plotDataOverTime2Display
    plotDataOverTime2Display.SeriesVisibility = ['Q (id=0)']
    #### saving camera placements for all active views

    ## current camera placement for renderView1
    #renderView1.CameraPosition = [0.04756324723884351, 0.013469162680682566, -0.00013035383570759377]
    #renderView1.CameraFocalPoint = [0.03385898657143116, 0.009294988121837378, -3.874010872095823e-05]
    #renderView1.CameraViewUp = [0.2865556377187659, -0.9443391555224921, -0.16158411381895746]
    #renderView1.CameraParallelScale = 0.003745658037500837


    # save data
    SaveData('/home/sansomk/caseFiles/mri/VWI_proj/cumulative/{0}_inlet.csv'.format(case), proxy=plotDataOverTime2, Precision=15,
        UseScientificNotation=1,
        WriteTimeSteps=0)
