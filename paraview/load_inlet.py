# trace generated using paraview version 5.5.0

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'XML Unstructured Grid Reader'
inlet_outfile_nodevtu = XMLUnstructuredGridReader(FileName=['/raid/sansomk/caseFiles/mri/VWI_proj/case1/fluent_dsa/vtk_out/inlet_outfile_node.vtu'])
inlet_outfile_nodevtu.PointArrayStatus = [ 'velocity']

# get the material library
materialLibrary1 = GetMaterialLibrary()

# Create a new 'Render View'
renderView1 = CreateView('RenderView')
renderView1.ViewSize = [1958, 1123]
renderView1.AxesGrid = 'GridAxes3DActor'
renderView1.StereoType = 0
renderView1.Background = [0.32, 0.34, 0.43]
renderView1.OSPRayMaterialLibrary = materialLibrary1

# init the 'GridAxes3DActor' selected for 'AxesGrid'
renderView1.AxesGrid.XTitleFontFile = ''
renderView1.AxesGrid.YTitleFontFile = ''
renderView1.AxesGrid.ZTitleFontFile = ''
renderView1.AxesGrid.XLabelFontFile = ''
renderView1.AxesGrid.YLabelFontFile = ''
renderView1.AxesGrid.ZLabelFontFile = ''

# get layout
layout1 = GetLayout()

# place view in the layout
layout1.AssignView(0, renderView1)

# show data in view
inlet_outfile_nodevtuDisplay = Show(inlet_outfile_nodevtu, renderView1)

# get color transfer function/color map for 'z_wall_shear'
velocity = GetColorTransferFunction('velocity')

# get opacity transfer function/opacity map for 'z_wall_shear'
z_wall_shearPWF = GetOpacityTransferFunction('velocity')

# trace defaults for the display properties.
inlet_outfile_nodevtuDisplay.Representation = 'Surface'
inlet_outfile_nodevtuDisplay.ColorArrayName = ['POINTS', 'velocity']
inlet_outfile_nodevtuDisplay.LookupTable = z_wall_shearLUT
inlet_outfile_nodevtuDisplay.OSPRayScaleArray = 'velocity'
inlet_outfile_nodevtuDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
inlet_outfile_nodevtuDisplay.SelectOrientationVectors = 'velocity'
inlet_outfile_nodevtuDisplay.ScaleFactor = 0.0004425810417160392
inlet_outfile_nodevtuDisplay.SelectScaleArray = 'velocity'
inlet_outfile_nodevtuDisplay.GlyphType = 'Arrow'
inlet_outfile_nodevtuDisplay.GlyphTableIndexArray = 'velocity'
inlet_outfile_nodevtuDisplay.GaussianRadius = 2.212905208580196e-05
inlet_outfile_nodevtuDisplay.SetScaleArray = ['POINTS', 'velocity']
inlet_outfile_nodevtuDisplay.ScaleTransferFunction = 'PiecewiseFunction'
inlet_outfile_nodevtuDisplay.OpacityArray = ['POINTS', 'velocity']
inlet_outfile_nodevtuDisplay.OpacityTransferFunction = 'PiecewiseFunction'
inlet_outfile_nodevtuDisplay.DataAxesGrid = 'GridAxesRepresentation'
inlet_outfile_nodevtuDisplay.SelectionCellLabelFontFile = ''
inlet_outfile_nodevtuDisplay.SelectionPointLabelFontFile = ''
inlet_outfile_nodevtuDisplay.PolarAxes = 'PolarAxesRepresentation'
inlet_outfile_nodevtuDisplay.ScalarOpacityFunction = z_wall_shearPWF
inlet_outfile_nodevtuDisplay.ScalarOpacityUnitDistance = 0.0004281476805389677

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
inlet_outfile_nodevtuDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.09353560954332352, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
inlet_outfile_nodevtuDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.09353560954332352, 1.0, 0.5, 0.0]

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
inlet_outfile_nodevtuDisplay.DataAxesGrid.XTitleFontFile = ''
inlet_outfile_nodevtuDisplay.DataAxesGrid.YTitleFontFile = ''
inlet_outfile_nodevtuDisplay.DataAxesGrid.ZTitleFontFile = ''
inlet_outfile_nodevtuDisplay.DataAxesGrid.XLabelFontFile = ''
inlet_outfile_nodevtuDisplay.DataAxesGrid.YLabelFontFile = ''
inlet_outfile_nodevtuDisplay.DataAxesGrid.ZLabelFontFile = ''

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
inlet_outfile_nodevtuDisplay.PolarAxes.PolarAxisTitleFontFile = ''
inlet_outfile_nodevtuDisplay.PolarAxes.PolarAxisLabelFontFile = ''
inlet_outfile_nodevtuDisplay.PolarAxes.LastRadialAxisTextFontFile = ''
inlet_outfile_nodevtuDisplay.PolarAxes.SecondaryRadialAxesTextFontFile = ''

# reset view to fit data
renderView1.ResetCamera()

# show color bar/color legend
inlet_outfile_nodevtuDisplay.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Extract Surface'
extractSurface2 = ExtractSurface(Input=inlet_outfile_nodevtu)

# show data in view
extractSurface2Display = Show(extractSurface2, renderView1)

# trace defaults for the display properties.
extractSurface2Display.Representation = 'Surface'
extractSurface2Display.ColorArrayName = ['POINTS', 'velocity']
extractSurface2Display.LookupTable = z_wall_shearLUT
extractSurface2Display.OSPRayScaleArray = 'velocity'
extractSurface2Display.OSPRayScaleFunction = 'PiecewiseFunction'
extractSurface2Display.SelectOrientationVectors = 'velocity'
extractSurface2Display.ScaleFactor = 0.0004425810417160392
extractSurface2Display.SelectScaleArray = 'velocity'
extractSurface2Display.GlyphType = 'Arrow'
extractSurface2Display.GlyphTableIndexArray = 'z_wall_shear'
extractSurface2Display.GaussianRadius = 2.212905208580196e-05
extractSurface2Display.SetScaleArray = ['POINTS', 'velocity']
extractSurface2Display.ScaleTransferFunction = 'PiecewiseFunction'
extractSurface2Display.OpacityArray = ['POINTS', 'velocity']
extractSurface2Display.OpacityTransferFunction = 'PiecewiseFunction'
extractSurface2Display.DataAxesGrid = 'GridAxesRepresentation'
extractSurface2Display.SelectionCellLabelFontFile = ''
extractSurface2Display.SelectionPointLabelFontFile = ''
extractSurface2Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
extractSurface2Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.09353560954332352, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
extractSurface2Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.09353560954332352, 1.0, 0.5, 0.0]

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
extractSurface2Display.DataAxesGrid.XTitleFontFile = ''
extractSurface2Display.DataAxesGrid.YTitleFontFile = ''
extractSurface2Display.DataAxesGrid.ZTitleFontFile = ''
extractSurface2Display.DataAxesGrid.XLabelFontFile = ''
extractSurface2Display.DataAxesGrid.YLabelFontFile = ''
extractSurface2Display.DataAxesGrid.ZLabelFontFile = ''

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
extractSurface2Display.PolarAxes.PolarAxisTitleFontFile = ''
extractSurface2Display.PolarAxes.PolarAxisLabelFontFile = ''
extractSurface2Display.PolarAxes.LastRadialAxisTextFontFile = ''
extractSurface2Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''

# hide data in view
Hide(inlet_outfile_nodevtu, renderView1)

# show color bar/color legend
extractSurface2Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Generate Surface Normals'
generateSurfaceNormals2 = GenerateSurfaceNormals(Input=extractSurface2)

# show data in view
generateSurfaceNormals2Display = Show(generateSurfaceNormals2, renderView1)

# trace defaults for the display properties.
generateSurfaceNormals2Display.Representation = 'Surface'
generateSurfaceNormals2Display.ColorArrayName = ['POINTS', 'z_wall_shear']
generateSurfaceNormals2Display.LookupTable = z_wall_shearLUT
generateSurfaceNormals2Display.OSPRayScaleArray = 'z_wall_shear'
generateSurfaceNormals2Display.OSPRayScaleFunction = 'PiecewiseFunction'
generateSurfaceNormals2Display.SelectOrientationVectors = 'velocity'
generateSurfaceNormals2Display.ScaleFactor = 0.0004425810417160392
generateSurfaceNormals2Display.SelectScaleArray = 'z_wall_shear'
generateSurfaceNormals2Display.GlyphType = 'Arrow'
generateSurfaceNormals2Display.GlyphTableIndexArray = 'z_wall_shear'
generateSurfaceNormals2Display.GaussianRadius = 2.212905208580196e-05
generateSurfaceNormals2Display.SetScaleArray = ['POINTS', 'z_wall_shear']
generateSurfaceNormals2Display.ScaleTransferFunction = 'PiecewiseFunction'
generateSurfaceNormals2Display.OpacityArray = ['POINTS', 'z_wall_shear']
generateSurfaceNormals2Display.OpacityTransferFunction = 'PiecewiseFunction'
generateSurfaceNormals2Display.DataAxesGrid = 'GridAxesRepresentation'
generateSurfaceNormals2Display.SelectionCellLabelFontFile = ''
generateSurfaceNormals2Display.SelectionPointLabelFontFile = ''
generateSurfaceNormals2Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
generateSurfaceNormals2Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.09353560954332352, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
generateSurfaceNormals2Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.09353560954332352, 1.0, 0.5, 0.0]

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
generateSurfaceNormals2Display.DataAxesGrid.XTitleFontFile = ''
generateSurfaceNormals2Display.DataAxesGrid.YTitleFontFile = ''
generateSurfaceNormals2Display.DataAxesGrid.ZTitleFontFile = ''
generateSurfaceNormals2Display.DataAxesGrid.XLabelFontFile = ''
generateSurfaceNormals2Display.DataAxesGrid.YLabelFontFile = ''
generateSurfaceNormals2Display.DataAxesGrid.ZLabelFontFile = ''

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
generateSurfaceNormals2Display.PolarAxes.PolarAxisTitleFontFile = ''
generateSurfaceNormals2Display.PolarAxes.PolarAxisLabelFontFile = ''
generateSurfaceNormals2Display.PolarAxes.LastRadialAxisTextFontFile = ''
generateSurfaceNormals2Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''

# hide data in view
Hide(extractSurface2, renderView1)

# show color bar/color legend
generateSurfaceNormals2Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Calculator'
calculator4 = Calculator(Input=generateSurfaceNormals2)
calculator4.Function = ''

# set active source
SetActiveSource(calculator2)

# set active source
SetActiveSource(calculator1)

# set active source
SetActiveSource(calculator4)

# Properties modified on calculator4
calculator4.ResultArrayName = 'vdotn'
calculator4.Function = 'velocity_X*-Normals_X+velocity_Y*-Normals_Y+velocity_Z*-Normals_Z'

# show data in view
calculator4Display = Show(calculator4, renderView1)

# get color transfer function/color map for 'vdotn'
vdotnLUT = GetColorTransferFunction('vdotn')

# trace defaults for the display properties.
calculator4Display.Representation = 'Surface'
calculator4Display.ColorArrayName = ['POINTS', 'vdotn']
calculator4Display.LookupTable = vdotnLUT
calculator4Display.OSPRayScaleArray = 'vdotn'
calculator4Display.OSPRayScaleFunction = 'PiecewiseFunction'
calculator4Display.SelectOrientationVectors = 'velocity'
calculator4Display.ScaleFactor = 0.0004425810417160392
calculator4Display.SelectScaleArray = 'vdotn'
calculator4Display.GlyphType = 'Arrow'
calculator4Display.GlyphTableIndexArray = 'vdotn'
calculator4Display.GaussianRadius = 2.212905208580196e-05
calculator4Display.SetScaleArray = ['POINTS', 'vdotn']
calculator4Display.ScaleTransferFunction = 'PiecewiseFunction'
calculator4Display.OpacityArray = ['POINTS', 'vdotn']
calculator4Display.OpacityTransferFunction = 'PiecewiseFunction'
calculator4Display.DataAxesGrid = 'GridAxesRepresentation'
calculator4Display.SelectionCellLabelFontFile = ''
calculator4Display.SelectionPointLabelFontFile = ''
calculator4Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
calculator4Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.3393737108883882, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
calculator4Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.3393737108883882, 1.0, 0.5, 0.0]

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
calculator4Display.DataAxesGrid.XTitleFontFile = ''
calculator4Display.DataAxesGrid.YTitleFontFile = ''
calculator4Display.DataAxesGrid.ZTitleFontFile = ''
calculator4Display.DataAxesGrid.XLabelFontFile = ''
calculator4Display.DataAxesGrid.YLabelFontFile = ''
calculator4Display.DataAxesGrid.ZLabelFontFile = ''

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
calculator4Display.PolarAxes.PolarAxisTitleFontFile = ''
calculator4Display.PolarAxes.PolarAxisLabelFontFile = ''
calculator4Display.PolarAxes.LastRadialAxisTextFontFile = ''
calculator4Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''

# hide data in view
Hide(generateSurfaceNormals2, renderView1)

# show color bar/color legend
calculator4Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Integrate Variables'
integrateVariables2 = IntegrateVariables(Input=calculator4)

# Create a new 'SpreadSheet View'
spreadSheetView1 = CreateView('SpreadSheetView')
spreadSheetView1.ColumnToSort = ''
spreadSheetView1.BlockSize = 1024L
# uncomment following to set a specific view size
# spreadSheetView1.ViewSize = [400, 400]

# place view in the layout
layout1.AssignView(2, spreadSheetView1)

# show data in view
integrateVariables2Display = Show(integrateVariables2, spreadSheetView1)

# update the view to ensure updated data information
spreadSheetView1.Update()

# create a new 'Calculator'
calculator5 = Calculator(Input=integrateVariables2)
calculator5.Function = ''

# set active source
SetActiveSource(calculator2)

# set active source
SetActiveSource(integrateVariables2)

# set active source
SetActiveSource(calculator5)

# Properties modified on calculator5
calculator5.ResultArrayName = 'Q'
calculator5.Function = 'vdotn*10^6*60'

# show data in view
calculator5Display = Show(calculator5, spreadSheetView1)

# hide data in view
Hide(integrateVariables2, spreadSheetView1)

# update the view to ensure updated data information
spreadSheetView1.Update()

# create a new 'Calculator'
calculator6 = Calculator(Input=calculator5)
calculator6.Function = ''

# set active source
SetActiveSource(calculator3)

# set active source
SetActiveSource(calculator6)

# set active source
SetActiveSource(calculator4)

# set active source
SetActiveSource(calculator3)

# set active source
SetActiveSource(calculator6)

# Properties modified on calculator6
calculator6.ResultArrayName = 'Re'
calculator6.Function = 'vdotn/1.54330872788623e-5*1050.0/0.0035*0.00443283397'

# show data in view
calculator6Display = Show(calculator6, spreadSheetView1)

# hide data in view
Hide(calculator5, spreadSheetView1)

# update the view to ensure updated data information
spreadSheetView1.Update()

# create a new 'Plot Data Over Time'
plotDataOverTime2 = PlotDataOverTime(Input=calculator6)

# Create a new 'Quartile Chart View'
quartileChartView1 = CreateView('QuartileChartView')
quartileChartView1.ViewSize = [974, 546]
quartileChartView1.ChartTitleFontFile = ''
quartileChartView1.LeftAxisTitleFontFile = ''
quartileChartView1.LeftAxisRangeMaximum = 6.66
quartileChartView1.LeftAxisLabelFontFile = ''
quartileChartView1.BottomAxisTitleFontFile = ''
quartileChartView1.BottomAxisRangeMaximum = 6.66
quartileChartView1.BottomAxisLabelFontFile = ''
quartileChartView1.RightAxisRangeMaximum = 6.66
quartileChartView1.RightAxisLabelFontFile = ''
quartileChartView1.TopAxisTitleFontFile = ''
quartileChartView1.TopAxisRangeMaximum = 6.66
quartileChartView1.TopAxisLabelFontFile = ''

# place view in the layout
layout1.AssignView(6, quartileChartView1)

# show data in view
plotDataOverTime2Display = Show(plotDataOverTime2, quartileChartView1)

# trace defaults for the display properties.
plotDataOverTime2Display.AttributeType = 'Row Data'
plotDataOverTime2Display.UseIndexForXAxis = 0
plotDataOverTime2Display.XArrayName = 'Time'
plotDataOverTime2Display.SeriesVisibility = ['absolute_pressure (stats)', 'Normals (Magnitude) (stats)', 'Q (stats)', 'Re (stats)', 'vdotn (stats)', 'velocity (Magnitude) (stats)', 'X (stats)', 'x_velocity (stats)', 'x_wall_shear (stats)', 'Y (stats)', 'y_velocity (stats)', 'y_wall_shear (stats)', 'Z (stats)', 'z_velocity (stats)', 'z_wall_shear (stats)']
plotDataOverTime2Display.SeriesLabel = ['absolute_pressure (stats)', 'absolute_pressure (stats)', 'Normals (0) (stats)', 'Normals (0) (stats)', 'Normals (1) (stats)', 'Normals (1) (stats)', 'Normals (2) (stats)', 'Normals (2) (stats)', 'Normals (Magnitude) (stats)', 'Normals (Magnitude) (stats)', 'Q (stats)', 'Q (stats)', 'Re (stats)', 'Re (stats)', 'vdotn (stats)', 'vdotn (stats)', 'velocity (0) (stats)', 'velocity (0) (stats)', 'velocity (1) (stats)', 'velocity (1) (stats)', 'velocity (2) (stats)', 'velocity (2) (stats)', 'velocity (Magnitude) (stats)', 'velocity (Magnitude) (stats)', 'X (stats)', 'X (stats)', 'x_velocity (stats)', 'x_velocity (stats)', 'x_wall_shear (stats)', 'x_wall_shear (stats)', 'Y (stats)', 'Y (stats)', 'y_velocity (stats)', 'y_velocity (stats)', 'y_wall_shear (stats)', 'y_wall_shear (stats)', 'Z (stats)', 'Z (stats)', 'z_velocity (stats)', 'z_velocity (stats)', 'z_wall_shear (stats)', 'z_wall_shear (stats)', 'N (stats)', 'N (stats)', 'Time (stats)', 'Time (stats)', 'vtkValidPointMask (stats)', 'vtkValidPointMask (stats)']
plotDataOverTime2Display.SeriesColor = ['absolute_pressure (stats)', '0', '0', '0', 'Normals (0) (stats)', '0.89', '0.1', '0.11', 'Normals (1) (stats)', '0.22', '0.49', '0.72', 'Normals (2) (stats)', '0.3', '0.69', '0.29', 'Normals (Magnitude) (stats)', '0.6', '0.31', '0.64', 'Q (stats)', '1', '0.5', '0', 'Re (stats)', '0.65', '0.34', '0.16', 'vdotn (stats)', '0', '0', '0', 'velocity (0) (stats)', '0.89', '0.1', '0.11', 'velocity (1) (stats)', '0.22', '0.49', '0.72', 'velocity (2) (stats)', '0.3', '0.69', '0.29', 'velocity (Magnitude) (stats)', '0.6', '0.31', '0.64', 'X (stats)', '1', '0.5', '0', 'x_velocity (stats)', '0.65', '0.34', '0.16', 'x_wall_shear (stats)', '0', '0', '0', 'Y (stats)', '0.89', '0.1', '0.11', 'y_velocity (stats)', '0.22', '0.49', '0.72', 'y_wall_shear (stats)', '0.3', '0.69', '0.29', 'Z (stats)', '0.6', '0.31', '0.64', 'z_velocity (stats)', '1', '0.5', '0', 'z_wall_shear (stats)', '0.65', '0.34', '0.16', 'N (stats)', '0', '0', '0', 'Time (stats)', '0.89', '0.1', '0.11', 'vtkValidPointMask (stats)', '0.22', '0.49', '0.72']
plotDataOverTime2Display.SeriesPlotCorner = ['absolute_pressure (stats)', '0', 'Normals (0) (stats)', '0', 'Normals (1) (stats)', '0', 'Normals (2) (stats)', '0', 'Normals (Magnitude) (stats)', '0', 'Q (stats)', '0', 'Re (stats)', '0', 'vdotn (stats)', '0', 'velocity (0) (stats)', '0', 'velocity (1) (stats)', '0', 'velocity (2) (stats)', '0', 'velocity (Magnitude) (stats)', '0', 'X (stats)', '0', 'x_velocity (stats)', '0', 'x_wall_shear (stats)', '0', 'Y (stats)', '0', 'y_velocity (stats)', '0', 'y_wall_shear (stats)', '0', 'Z (stats)', '0', 'z_velocity (stats)', '0', 'z_wall_shear (stats)', '0', 'N (stats)', '0', 'Time (stats)', '0', 'vtkValidPointMask (stats)', '0']
plotDataOverTime2Display.SeriesLabelPrefix = ''
plotDataOverTime2Display.SeriesLineStyle = ['absolute_pressure (stats)', '1', 'Normals (0) (stats)', '1', 'Normals (1) (stats)', '1', 'Normals (2) (stats)', '1', 'Normals (Magnitude) (stats)', '1', 'Q (stats)', '1', 'Re (stats)', '1', 'vdotn (stats)', '1', 'velocity (0) (stats)', '1', 'velocity (1) (stats)', '1', 'velocity (2) (stats)', '1', 'velocity (Magnitude) (stats)', '1', 'X (stats)', '1', 'x_velocity (stats)', '1', 'x_wall_shear (stats)', '1', 'Y (stats)', '1', 'y_velocity (stats)', '1', 'y_wall_shear (stats)', '1', 'Z (stats)', '1', 'z_velocity (stats)', '1', 'z_wall_shear (stats)', '1', 'N (stats)', '1', 'Time (stats)', '1', 'vtkValidPointMask (stats)', '1']
plotDataOverTime2Display.SeriesLineThickness = ['absolute_pressure (stats)', '2', 'Normals (0) (stats)', '2', 'Normals (1) (stats)', '2', 'Normals (2) (stats)', '2', 'Normals (Magnitude) (stats)', '2', 'Q (stats)', '2', 'Re (stats)', '2', 'vdotn (stats)', '2', 'velocity (0) (stats)', '2', 'velocity (1) (stats)', '2', 'velocity (2) (stats)', '2', 'velocity (Magnitude) (stats)', '2', 'X (stats)', '2', 'x_velocity (stats)', '2', 'x_wall_shear (stats)', '2', 'Y (stats)', '2', 'y_velocity (stats)', '2', 'y_wall_shear (stats)', '2', 'Z (stats)', '2', 'z_velocity (stats)', '2', 'z_wall_shear (stats)', '2', 'N (stats)', '2', 'Time (stats)', '2', 'vtkValidPointMask (stats)', '2']
plotDataOverTime2Display.SeriesMarkerStyle = ['absolute_pressure (stats)', '0', 'Normals (0) (stats)', '0', 'Normals (1) (stats)', '0', 'Normals (2) (stats)', '0', 'Normals (Magnitude) (stats)', '0', 'Q (stats)', '0', 'Re (stats)', '0', 'vdotn (stats)', '0', 'velocity (0) (stats)', '0', 'velocity (1) (stats)', '0', 'velocity (2) (stats)', '0', 'velocity (Magnitude) (stats)', '0', 'X (stats)', '0', 'x_velocity (stats)', '0', 'x_wall_shear (stats)', '0', 'Y (stats)', '0', 'y_velocity (stats)', '0', 'y_wall_shear (stats)', '0', 'Z (stats)', '0', 'z_velocity (stats)', '0', 'z_wall_shear (stats)', '0', 'N (stats)', '0', 'Time (stats)', '0', 'vtkValidPointMask (stats)', '0']

# update the view to ensure updated data information
spreadSheetView1.Update()

# update the view to ensure updated data information
quartileChartView1.Update()

# set active view
SetActiveView(renderView1)

# set active view
SetActiveView(quartileChartView1)

# Properties modified on plotDataOverTime2Display
plotDataOverTime2Display.SeriesVisibility = ['absolute_pressure (stats)', 'Q (stats)', 'Re (stats)', 'vdotn (stats)', 'velocity (Magnitude) (stats)', 'X (stats)', 'x_velocity (stats)', 'x_wall_shear (stats)', 'Y (stats)', 'y_velocity (stats)', 'y_wall_shear (stats)', 'Z (stats)', 'z_velocity (stats)', 'z_wall_shear (stats)']
plotDataOverTime2Display.SeriesColor = ['absolute_pressure (stats)', '0', '0', '0', 'Normals (0) (stats)', '0.889998', '0.100008', '0.110002', 'Normals (1) (stats)', '0.220005', '0.489998', '0.719997', 'Normals (2) (stats)', '0.300008', '0.689998', '0.289998', 'Normals (Magnitude) (stats)', '0.6', '0.310002', '0.639994', 'Q (stats)', '1', '0.500008', '0', 'Re (stats)', '0.650004', '0.340002', '0.160006', 'vdotn (stats)', '0', '0', '0', 'velocity (0) (stats)', '0.889998', '0.100008', '0.110002', 'velocity (1) (stats)', '0.220005', '0.489998', '0.719997', 'velocity (2) (stats)', '0.300008', '0.689998', '0.289998', 'velocity (Magnitude) (stats)', '0.6', '0.310002', '0.639994', 'X (stats)', '1', '0.500008', '0', 'x_velocity (stats)', '0.650004', '0.340002', '0.160006', 'x_wall_shear (stats)', '0', '0', '0', 'Y (stats)', '0.889998', '0.100008', '0.110002', 'y_velocity (stats)', '0.220005', '0.489998', '0.719997', 'y_wall_shear (stats)', '0.300008', '0.689998', '0.289998', 'Z (stats)', '0.6', '0.310002', '0.639994', 'z_velocity (stats)', '1', '0.500008', '0', 'z_wall_shear (stats)', '0.650004', '0.340002', '0.160006', 'N (stats)', '0', '0', '0', 'Time (stats)', '0.889998', '0.100008', '0.110002', 'vtkValidPointMask (stats)', '0.220005', '0.489998', '0.719997']
plotDataOverTime2Display.SeriesPlotCorner = ['N (stats)', '0', 'Normals (0) (stats)', '0', 'Normals (1) (stats)', '0', 'Normals (2) (stats)', '0', 'Normals (Magnitude) (stats)', '0', 'Q (stats)', '0', 'Re (stats)', '0', 'Time (stats)', '0', 'X (stats)', '0', 'Y (stats)', '0', 'Z (stats)', '0', 'absolute_pressure (stats)', '0', 'vdotn (stats)', '0', 'velocity (0) (stats)', '0', 'velocity (1) (stats)', '0', 'velocity (2) (stats)', '0', 'velocity (Magnitude) (stats)', '0', 'vtkValidPointMask (stats)', '0', 'x_velocity (stats)', '0', 'x_wall_shear (stats)', '0', 'y_velocity (stats)', '0', 'y_wall_shear (stats)', '0', 'z_velocity (stats)', '0', 'z_wall_shear (stats)', '0']
plotDataOverTime2Display.SeriesLineStyle = ['N (stats)', '1', 'Normals (0) (stats)', '1', 'Normals (1) (stats)', '1', 'Normals (2) (stats)', '1', 'Normals (Magnitude) (stats)', '1', 'Q (stats)', '1', 'Re (stats)', '1', 'Time (stats)', '1', 'X (stats)', '1', 'Y (stats)', '1', 'Z (stats)', '1', 'absolute_pressure (stats)', '1', 'vdotn (stats)', '1', 'velocity (0) (stats)', '1', 'velocity (1) (stats)', '1', 'velocity (2) (stats)', '1', 'velocity (Magnitude) (stats)', '1', 'vtkValidPointMask (stats)', '1', 'x_velocity (stats)', '1', 'x_wall_shear (stats)', '1', 'y_velocity (stats)', '1', 'y_wall_shear (stats)', '1', 'z_velocity (stats)', '1', 'z_wall_shear (stats)', '1']
plotDataOverTime2Display.SeriesLineThickness = ['N (stats)', '2', 'Normals (0) (stats)', '2', 'Normals (1) (stats)', '2', 'Normals (2) (stats)', '2', 'Normals (Magnitude) (stats)', '2', 'Q (stats)', '2', 'Re (stats)', '2', 'Time (stats)', '2', 'X (stats)', '2', 'Y (stats)', '2', 'Z (stats)', '2', 'absolute_pressure (stats)', '2', 'vdotn (stats)', '2', 'velocity (0) (stats)', '2', 'velocity (1) (stats)', '2', 'velocity (2) (stats)', '2', 'velocity (Magnitude) (stats)', '2', 'vtkValidPointMask (stats)', '2', 'x_velocity (stats)', '2', 'x_wall_shear (stats)', '2', 'y_velocity (stats)', '2', 'y_wall_shear (stats)', '2', 'z_velocity (stats)', '2', 'z_wall_shear (stats)', '2']
plotDataOverTime2Display.SeriesMarkerStyle = ['N (stats)', '0', 'Normals (0) (stats)', '0', 'Normals (1) (stats)', '0', 'Normals (2) (stats)', '0', 'Normals (Magnitude) (stats)', '0', 'Q (stats)', '0', 'Re (stats)', '0', 'Time (stats)', '0', 'X (stats)', '0', 'Y (stats)', '0', 'Z (stats)', '0', 'absolute_pressure (stats)', '0', 'vdotn (stats)', '0', 'velocity (0) (stats)', '0', 'velocity (1) (stats)', '0', 'velocity (2) (stats)', '0', 'velocity (Magnitude) (stats)', '0', 'vtkValidPointMask (stats)', '0', 'x_velocity (stats)', '0', 'x_wall_shear (stats)', '0', 'y_velocity (stats)', '0', 'y_wall_shear (stats)', '0', 'z_velocity (stats)', '0', 'z_wall_shear (stats)', '0']

# Properties modified on plotDataOverTime2Display
plotDataOverTime2Display.SeriesVisibility = ['Q (stats)', 'Re (stats)', 'vdotn (stats)', 'velocity (Magnitude) (stats)', 'X (stats)', 'x_velocity (stats)', 'x_wall_shear (stats)', 'Y (stats)', 'y_velocity (stats)', 'y_wall_shear (stats)', 'Z (stats)', 'z_velocity (stats)', 'z_wall_shear (stats)']

# Properties modified on plotDataOverTime2Display
plotDataOverTime2Display.SeriesVisibility = ['Re (stats)', 'vdotn (stats)', 'velocity (Magnitude) (stats)', 'X (stats)', 'x_velocity (stats)', 'x_wall_shear (stats)', 'Y (stats)', 'y_velocity (stats)', 'y_wall_shear (stats)', 'Z (stats)', 'z_velocity (stats)', 'z_wall_shear (stats)']

# Properties modified on plotDataOverTime2Display
plotDataOverTime2Display.SeriesVisibility = ['vdotn (stats)', 'velocity (Magnitude) (stats)', 'X (stats)', 'x_velocity (stats)', 'x_wall_shear (stats)', 'Y (stats)', 'y_velocity (stats)', 'y_wall_shear (stats)', 'Z (stats)', 'z_velocity (stats)', 'z_wall_shear (stats)']

# Properties modified on plotDataOverTime2Display
plotDataOverTime2Display.SeriesVisibility = ['Q (stats)', 'vdotn (stats)', 'velocity (Magnitude) (stats)', 'X (stats)', 'x_velocity (stats)', 'x_wall_shear (stats)', 'Y (stats)', 'y_velocity (stats)', 'y_wall_shear (stats)', 'Z (stats)', 'z_velocity (stats)', 'z_wall_shear (stats)']

# Properties modified on plotDataOverTime2Display
plotDataOverTime2Display.SeriesVisibility = ['Q (stats)', 'velocity (Magnitude) (stats)', 'X (stats)', 'x_velocity (stats)', 'x_wall_shear (stats)', 'Y (stats)', 'y_velocity (stats)', 'y_wall_shear (stats)', 'Z (stats)', 'z_velocity (stats)', 'z_wall_shear (stats)']

# Properties modified on plotDataOverTime2Display
plotDataOverTime2Display.SeriesVisibility = ['Q (stats)', 'X (stats)', 'x_velocity (stats)', 'x_wall_shear (stats)', 'Y (stats)', 'y_velocity (stats)', 'y_wall_shear (stats)', 'Z (stats)', 'z_velocity (stats)', 'z_wall_shear (stats)']

# Properties modified on plotDataOverTime2Display
plotDataOverTime2Display.SeriesVisibility = ['Q (stats)', 'X (stats)', 'x_wall_shear (stats)', 'Y (stats)', 'y_velocity (stats)', 'y_wall_shear (stats)', 'Z (stats)', 'z_velocity (stats)', 'z_wall_shear (stats)']

# Properties modified on plotDataOverTime2Display
plotDataOverTime2Display.SeriesVisibility = ['Q (stats)', 'X (stats)', 'x_wall_shear (stats)', 'y_velocity (stats)', 'y_wall_shear (stats)', 'Z (stats)', 'z_velocity (stats)', 'z_wall_shear (stats)']

# Properties modified on plotDataOverTime2Display
plotDataOverTime2Display.SeriesVisibility = ['Q (stats)', 'X (stats)', 'x_wall_shear (stats)', 'y_velocity (stats)', 'y_wall_shear (stats)', 'z_velocity (stats)', 'z_wall_shear (stats)']

# Properties modified on plotDataOverTime2Display
plotDataOverTime2Display.SeriesVisibility = ['Q (stats)', 'X (stats)', 'x_wall_shear (stats)', 'y_velocity (stats)', 'y_wall_shear (stats)', 'z_wall_shear (stats)']

# Properties modified on plotDataOverTime2Display
plotDataOverTime2Display.SeriesVisibility = ['Q (stats)', 'X (stats)', 'x_wall_shear (stats)', 'y_velocity (stats)', 'y_wall_shear (stats)']

# Properties modified on plotDataOverTime2Display
plotDataOverTime2Display.SeriesVisibility = ['Q (stats)', 'X (stats)', 'x_wall_shear (stats)', 'y_velocity (stats)']

# Properties modified on plotDataOverTime2Display
plotDataOverTime2Display.SeriesVisibility = ['Q (stats)', 'X (stats)', 'x_wall_shear (stats)']

# Properties modified on plotDataOverTime2Display
plotDataOverTime2Display.SeriesVisibility = ['Q (stats)', 'X (stats)']

# Properties modified on plotDataOverTime2Display
plotDataOverTime2Display.SeriesVisibility = ['Q (stats)']

#### saving camera placements for all active views

# current camera placement for renderView1
renderView1.CameraPosition = [0.023720674031015507, 0.019348313536698802, 0.0032664549126214795]
renderView1.CameraFocalPoint = [0.03385898657143116, 0.009294988121837378, -3.874010872095751e-05]
renderView1.CameraViewUp = [0.40670237541763354, 0.6282068509462543, -0.663286763213201]
renderView1.CameraParallelScale = 0.0031347781858912585

#### uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).
