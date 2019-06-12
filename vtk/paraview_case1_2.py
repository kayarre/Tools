#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'XML Unstructured Grid Reader'
calc_test_node_statsvtu = XMLUnstructuredGridReader(FileName=['/raid/home/ksansom/caseFiles/mri/VWI_proj/case1/fluent_dsa/vtk_out/calc_test_node_stats.vtu'])
calc_test_node_statsvtu.PointArrayStatus = ['WSS', 'WSSG', 'absolute_pressure', 'TAWSS', 'TAWSSG', 'OSI', 'velocity']

# create a new 'XML Unstructured Grid Reader'
calc_test_nodevtu = XMLUnstructuredGridReader(FileName=['/raid/home/ksansom/caseFiles/mri/VWI_proj/case1/fluent_dsa/vtk_out/calc_test_node.vtu'])
calc_test_nodevtu.PointArrayStatus = ['absolute_pressure', 'velocity', 'x_velocity', 'x_wall_shear', 'y_velocity', 'y_wall_shear', 'z_velocity', 'z_wall_shear', 'WSS', 'x_WSS_grad', 'y_WSS_grad', 'z_WSS_grad', 'WSSG']

# get animation scene
animationScene1 = GetAnimationScene()

# update animation scene based on data timesteps
animationScene1.UpdateAnimationUsingDataTimeSteps()

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
# renderView1.ViewSize = [1128, 649]

# get color transfer function/color map for 'OSI'
oSILUT = GetColorTransferFunction('OSI')

# get opacity transfer function/opacity map for 'OSI'
oSIPWF = GetOpacityTransferFunction('OSI')

# show data in view
calc_test_node_statsvtuDisplay = Show(calc_test_node_statsvtu, renderView1)
# trace defaults for the display properties.
calc_test_node_statsvtuDisplay.Representation = 'Surface'
calc_test_node_statsvtuDisplay.ColorArrayName = ['POINTS', 'OSI']
calc_test_node_statsvtuDisplay.LookupTable = oSILUT
calc_test_node_statsvtuDisplay.OSPRayScaleArray = 'OSI'
calc_test_node_statsvtuDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
calc_test_node_statsvtuDisplay.SelectOrientationVectors = 'velocity'
calc_test_node_statsvtuDisplay.ScaleFactor = 0.0033610027574468406
calc_test_node_statsvtuDisplay.SelectScaleArray = 'OSI'
calc_test_node_statsvtuDisplay.GlyphType = 'Arrow'
calc_test_node_statsvtuDisplay.GlyphTableIndexArray = 'OSI'
calc_test_node_statsvtuDisplay.DataAxesGrid = 'GridAxesRepresentation'
calc_test_node_statsvtuDisplay.PolarAxes = 'PolarAxesRepresentation'
calc_test_node_statsvtuDisplay.ScalarOpacityFunction = oSIPWF
calc_test_node_statsvtuDisplay.ScalarOpacityUnitDistance = 0.000833608401117992
calc_test_node_statsvtuDisplay.GaussianRadius = 0.0016805013787234203
calc_test_node_statsvtuDisplay.SetScaleArray = ['POINTS', 'OSI']
calc_test_node_statsvtuDisplay.ScaleTransferFunction = 'PiecewiseFunction'
calc_test_node_statsvtuDisplay.OpacityArray = ['POINTS', 'OSI']
calc_test_node_statsvtuDisplay.OpacityTransferFunction = 'PiecewiseFunction'

# reset view to fit data
renderView1.ResetCamera()

# show color bar/color legend
calc_test_node_statsvtuDisplay.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'WSSG'
wSSGLUT = GetColorTransferFunction('WSSG')

# get opacity transfer function/opacity map for 'WSSG'
wSSGPWF = GetOpacityTransferFunction('WSSG')

# show data in view
calc_test_nodevtuDisplay = Show(calc_test_nodevtu, renderView1)
# trace defaults for the display properties.
calc_test_nodevtuDisplay.Representation = 'Surface'
calc_test_nodevtuDisplay.ColorArrayName = ['POINTS', 'WSSG']
calc_test_nodevtuDisplay.LookupTable = wSSGLUT
calc_test_nodevtuDisplay.OSPRayScaleArray = 'WSSG'
calc_test_nodevtuDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
calc_test_nodevtuDisplay.SelectOrientationVectors = 'velocity'
calc_test_nodevtuDisplay.ScaleFactor = 0.0033610027574468406
calc_test_nodevtuDisplay.SelectScaleArray = 'WSSG'
calc_test_nodevtuDisplay.GlyphType = 'Arrow'
calc_test_nodevtuDisplay.GlyphTableIndexArray = 'WSSG'
calc_test_nodevtuDisplay.DataAxesGrid = 'GridAxesRepresentation'
calc_test_nodevtuDisplay.PolarAxes = 'PolarAxesRepresentation'
calc_test_nodevtuDisplay.ScalarOpacityFunction = wSSGPWF
calc_test_nodevtuDisplay.ScalarOpacityUnitDistance = 0.000833608401117992
calc_test_nodevtuDisplay.GaussianRadius = 0.0016805013787234203
calc_test_nodevtuDisplay.SetScaleArray = ['POINTS', 'WSSG']
calc_test_nodevtuDisplay.ScaleTransferFunction = 'PiecewiseFunction'
calc_test_nodevtuDisplay.OpacityArray = ['POINTS', 'WSSG']
calc_test_nodevtuDisplay.OpacityTransferFunction = 'PiecewiseFunction'

# show color bar/color legend
calc_test_nodevtuDisplay.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# hide data in view
Hide(calc_test_nodevtu, renderView1)

# hide data in view
Hide(calc_test_node_statsvtu, renderView1)

# set active source
SetActiveSource(calc_test_node_statsvtu)

# show data in view
calc_test_node_statsvtuDisplay = Show(calc_test_node_statsvtu, renderView1)

# show color bar/color legend
calc_test_node_statsvtuDisplay.SetScalarBarVisibility(renderView1, True)

# reset view to fit data
renderView1.ResetCamera()

# create a new 'Clip'
clip1 = Clip(Input=calc_test_node_statsvtu)
clip1.ClipType = 'Plane'
clip1.Scalars = ['POINTS', 'OSI']
clip1.Value = -37.98761330911797

# Properties modified on clip1.ClipType
clip1.ClipType.Origin = [0.01819323504077828, 0.018846831112735617, -0.011124459575281291]
clip1.ClipType.Normal = [-0.009594807941738455, 0.08226498057876676, 0.9965643043130415]

# show data in view
clip1Display = Show(clip1, renderView1)
# trace defaults for the display properties.
clip1Display.Representation = 'Surface'
clip1Display.ColorArrayName = ['POINTS', 'OSI']
clip1Display.LookupTable = oSILUT
clip1Display.OSPRayScaleArray = 'OSI'
clip1Display.OSPRayScaleFunction = 'PiecewiseFunction'
clip1Display.SelectOrientationVectors = 'velocity'
clip1Display.ScaleFactor = 0.003313942957902327
clip1Display.SelectScaleArray = 'OSI'
clip1Display.GlyphType = 'Arrow'
clip1Display.GlyphTableIndexArray = 'OSI'
clip1Display.DataAxesGrid = 'GridAxesRepresentation'
clip1Display.PolarAxes = 'PolarAxesRepresentation'
clip1Display.ScalarOpacityFunction = oSIPWF
clip1Display.ScalarOpacityUnitDistance = 0.0008591445100520482
clip1Display.GaussianRadius = 0.0016569714789511636
clip1Display.SetScaleArray = ['POINTS', 'OSI']
clip1Display.ScaleTransferFunction = 'PiecewiseFunction'
clip1Display.OpacityArray = ['POINTS', 'OSI']
clip1Display.OpacityTransferFunction = 'PiecewiseFunction'

# hide data in view
Hide(calc_test_node_statsvtu, renderView1)

# show color bar/color legend
clip1Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Clip'
clip2 = Clip(Input=clip1)
clip2.ClipType = 'Plane'
clip2.Scalars = ['POINTS', 'OSI']
clip2.Value = -1.5720264728727718

# Properties modified on clip2.ClipType
clip2.ClipType.Origin = [0.025560948791594172, 0.009813337603884373, -0.006648442576220549]
clip2.ClipType.Normal = [-0.9181855638842633, -0.09147876341107539, 0.38544377815618985]

# show data in view
clip2Display = Show(clip2, renderView1)
# trace defaults for the display properties.
clip2Display.Representation = 'Surface'
clip2Display.ColorArrayName = ['POINTS', 'OSI']
clip2Display.LookupTable = oSILUT
clip2Display.OSPRayScaleArray = 'OSI'
clip2Display.OSPRayScaleFunction = 'PiecewiseFunction'
clip2Display.SelectOrientationVectors = 'velocity'
clip2Display.ScaleFactor = 0.001187275303527713
clip2Display.SelectScaleArray = 'OSI'
clip2Display.GlyphType = 'Arrow'
clip2Display.GlyphTableIndexArray = 'OSI'
clip2Display.DataAxesGrid = 'GridAxesRepresentation'
clip2Display.PolarAxes = 'PolarAxesRepresentation'
clip2Display.ScalarOpacityFunction = oSIPWF
clip2Display.ScalarOpacityUnitDistance = 0.0005131964917439687
clip2Display.GaussianRadius = 0.0005936376517638565
clip2Display.SetScaleArray = ['POINTS', 'OSI']
clip2Display.ScaleTransferFunction = 'PiecewiseFunction'
clip2Display.OpacityArray = ['POINTS', 'OSI']
clip2Display.OpacityTransferFunction = 'PiecewiseFunction'

# hide data in view
Hide(clip1, renderView1)

# show color bar/color legend
clip2Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Clip'
clip3 = Clip(Input=clip2)
clip3.ClipType = 'Cylinder'
clip3.Scalars = ['POINTS', 'OSI']
clip3.Value = -1.5720264728727718

# Properties modified on clip4.ClipType
clip3.ClipType.Center = [0.021693128538859654, 0.008854472895131336, -0.003515594838044406]
clip3.ClipType.Axis = [0.73497073654434, -0.17219493099639058, -0.6558711170364333]
clip3.ClipType.Radius = 0.002739769215118061

# show data in view
clip3Display = Show(clip3, renderView1)
# trace defaults for the display properties.
clip3Display.Representation = 'Surface'
clip3Display.ColorArrayName = ['POINTS', 'OSI']
clip3Display.LookupTable = oSILUT
clip3Display.OSPRayScaleArray = 'OSI'
clip3Display.OSPRayScaleFunction = 'PiecewiseFunction'
clip3Display.SelectOrientationVectors = 'velocity'
clip3Display.ScaleFactor = 0.0017976811155676843
clip3Display.SelectScaleArray = 'OSI'
clip3Display.GlyphType = 'Arrow'
clip3Display.GlyphTableIndexArray = 'OSI'
clip3Display.DataAxesGrid = 'GridAxesRepresentation'
clip3Display.PolarAxes = 'PolarAxesRepresentation'
clip3Display.ScalarOpacityFunction = oSIPWF
clip3Display.ScalarOpacityUnitDistance = 0.0005328707593485771
clip3Display.GaussianRadius = 0.0008988405577838422
clip3Display.SetScaleArray = ['POINTS', 'OSI']
clip3Display.ScaleTransferFunction = 'PiecewiseFunction'
clip3Display.OpacityArray = ['POINTS', 'OSI']
clip3Display.OpacityTransferFunction = 'PiecewiseFunction'

# hide data in view
Hide(clip2, renderView1)

# show color bar/color legend
clip3Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()


# create a new 'Clip'
clip5 = Clip(Input=clip4)
clip5.ClipType = 'Plane'
clip5.Scalars = ['POINTS', 'OSI']
clip5.Value = -1.3222665777669622

# init the 'Plane' selected for 'ClipType'
clip5.ClipType.Origin = [0.019452753476798534, 0.010626763687469065, -0.005078122074337443]


# create a new 'Clip'
clip4 = Clip(Input=clip3)
clip4.ClipType = 'Plane'
clip4.Scalars = ['POINTS', 'OSI']
clip4.Value = -1.3222665777669622

clip4_1.ClipType.Origin = [0.02100160087287667, 0.009499056150381751, -0.0038770971414250426]
clip4_1.ClipType.Normal = [0.03595195418661695, 0.9985036974252484, -0.04120465044471869]


# toggle 3D widget visibility (only when running from the GUI)
Show3DWidgets(proxy=clip4_1.ClipType)


# update the view to ensure updated data information
renderView1.Update()

# create a new 'Clip'
clip5 = Clip(Input=clip4_1)
clip5.ClipType = 'Plane'
clip5.Scalars = ['POINTS', 'OSI']
clip5.Value = -0.9076142977771454

# init the 'Plane' selected for 'ClipType'
clip5.ClipType.Origin = [0.02181950241556091, 0.01362362918054505, -0.0029803540332201325]
clip5.ClipType.Normal = [0.15537731759943707, 0.1674796627738652, -0.9735545448164454]

# show data in view
clip5Display = Show(clip5, renderView1)
# trace defaults for the display properties.
clip5Display.Representation = 'Surface'
clip5Display.ColorArrayName = ['POINTS', 'OSI']
clip5Display.LookupTable = oSILUT
clip5Display.OSPRayScaleArray = 'OSI'
clip5Display.OSPRayScaleFunction = 'PiecewiseFunction'
clip5Display.SelectOrientationVectors = 'velocity'
clip5Display.ScaleFactor = 0.00015334729105234148
clip5Display.SelectScaleArray = 'OSI'
clip5Display.GlyphType = 'Arrow'
clip5Display.GlyphTableIndexArray = 'OSI'
clip5Display.DataAxesGrid = 'GridAxesRepresentation'
clip5Display.PolarAxes = 'PolarAxesRepresentation'
clip5Display.ScalarOpacityFunction = oSIPWF
clip5Display.ScalarOpacityUnitDistance = 8.831366592932063e-05
clip5Display.GaussianRadius = 7.667364552617074e-05
clip5Display.SetScaleArray = ['POINTS', 'OSI']
clip5Display.ScaleTransferFunction = 'PiecewiseFunction'
clip5Display.OpacityArray = ['POINTS', 'OSI']
clip5Display.OpacityTransferFunction = 'PiecewiseFunction'

# hide data in view
Hide(clip4_1, renderView1)

# show color bar/color legend
clip5Display.SetScalarBarVisibility(renderView1, True)


# update the view to ensure updated data information
renderView1.Update()

# hide color bar/color legend
clip5Display.SetScalarBarVisibility(renderView1, False)

# show color bar/color legend
clip5Display.SetScalarBarVisibility(renderView1, True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
oSILUT.ApplyPreset('Preset', True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
oSIPWF.ApplyPreset('Preset', True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
oSILUT.ApplyPreset('Preset', True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
oSIPWF.ApplyPreset('Preset', True)

# rescale color and/or opacity maps used to exactly fit the current data range
clip5Display.RescaleTransferFunctionToDataRange(False, True)

# Rescale transfer function
oSILUT.RescaleTransferFunction(0.0, 0.5)

# Rescale transfer function
oSIPWF.RescaleTransferFunction(0.0, 0.5)

# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=clip5.ClipType)

# set active source
SetActiveSource(calc_test_nodevtu)

# create a new 'Clip'
clip6 = Clip(Input=calc_test_nodevtu)
clip6.ClipType = 'Plane'
clip6.Scalars = ['POINTS', 'WSSG']
clip6.Value = 153954.0290375752

# init the 'Plane' selected for 'ClipType'
clip6.ClipType.Origin = clip1.ClipType.Origin
clip6.ClipType.Normal = clip1.ClipType.Normal

# set active source
SetActiveSource(clip1)

# set active source
SetActiveSource(clip6)

# show data in view
clip6Display = Show(clip6, renderView1)
# trace defaults for the display properties.
clip6Display.Representation = 'Surface'
clip6Display.ColorArrayName = ['POINTS', 'WSSG']
clip6Display.LookupTable = wSSGLUT
clip6Display.OSPRayScaleArray = 'WSSG'
clip6Display.OSPRayScaleFunction = 'PiecewiseFunction'
clip6Display.SelectOrientationVectors = 'velocity'
clip6Display.ScaleFactor = 0.002247811108827591
clip6Display.SelectScaleArray = 'WSSG'
clip6Display.GlyphType = 'Arrow'
clip6Display.GlyphTableIndexArray = 'WSSG'
clip6Display.DataAxesGrid = 'GridAxesRepresentation'
clip6Display.PolarAxes = 'PolarAxesRepresentation'
clip6Display.ScalarOpacityFunction = wSSGPWF
clip6Display.ScalarOpacityUnitDistance = 0.0005471223753043885
clip6Display.GaussianRadius = 0.0011239055544137956
clip6Display.SetScaleArray = ['POINTS', 'WSSG']
clip6Display.ScaleTransferFunction = 'PiecewiseFunction'
clip6Display.OpacityArray = ['POINTS', 'WSSG']
clip6Display.OpacityTransferFunction = 'PiecewiseFunction'

# hide data in view
Hide(calc_test_nodevtu, renderView1)

# show color bar/color legend
clip6Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# Rescale transfer function
oSILUT.RescaleTransferFunction(-2.30073062265, 0.5)

# Rescale transfer function
oSIPWF.RescaleTransferFunction(-2.30073062265, 0.5)

# create a new 'Clip'
clip7 = Clip(Input=clip6)
clip7.ClipType = 'Plane'
clip7.Scalars = ['POINTS', 'WSSG']
clip7.Value = 57862.67282051597

# init the 'Plane' selected for 'ClipType'
clip7.ClipType.Origin = clip2.ClipType.Origin
clip7.ClipType.Normal = clip2.ClipType.Normal

# set active source
SetActiveSource(clip2)

# set active source
SetActiveSource(clip7)

# show data in view
clip7Display = Show(clip7, renderView1)
# trace defaults for the display properties.
clip7Display.Representation = 'Surface'
clip7Display.ColorArrayName = ['POINTS', 'WSSG']
clip7Display.LookupTable = wSSGLUT
clip7Display.OSPRayScaleArray = 'WSSG'
clip7Display.OSPRayScaleFunction = 'PiecewiseFunction'
clip7Display.SelectOrientationVectors = 'velocity'
clip7Display.ScaleFactor = 0.0017976811155676843
clip7Display.SelectScaleArray = 'WSSG'
clip7Display.GlyphType = 'Arrow'
clip7Display.GlyphTableIndexArray = 'WSSG'
clip7Display.DataAxesGrid = 'GridAxesRepresentation'
clip7Display.PolarAxes = 'PolarAxesRepresentation'
clip7Display.ScalarOpacityFunction = wSSGPWF
clip7Display.ScalarOpacityUnitDistance = 0.0005348890283051247
clip7Display.GaussianRadius = 0.0008988405577838422
clip7Display.SetScaleArray = ['POINTS', 'WSSG']
clip7Display.ScaleTransferFunction = 'PiecewiseFunction'
clip7Display.OpacityArray = ['POINTS', 'WSSG']
clip7Display.OpacityTransferFunction = 'PiecewiseFunction'

# hide data in view
Hide(clip6, renderView1)

# show color bar/color legend
clip7Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Clip'
clip8 = Clip(Input=clip7)
clip8.ClipType = 'Cylinder'
clip8.Scalars = ['POINTS', 'OSI']
clip8.Value = -1.5720264728727718

# Properties modified on clip4.ClipType
clip8.ClipType.Center = clip3.ClipType.Center
clip8.ClipType.Axis = clip3.ClipType.Axis
clip8.ClipType.Radius = clip3.ClipType.Radius

# set active source
SetActiveSource(clip8)

# toggle 3D widget visibility (only when running from the GUI)
Show3DWidgets(proxy=clip8.ClipType)

# show data in view
clip8Display = Show(clip8, renderView1)
# trace defaults for the display properties.
clip8Display.Representation = 'Surface'
clip8Display.ColorArrayName = ['POINTS', 'WSSG']
clip8Display.LookupTable = wSSGLUT
clip8Display.OSPRayScaleArray = 'WSSG'
clip8Display.OSPRayScaleFunction = 'PiecewiseFunction'
clip8Display.SelectOrientationVectors = 'velocity'
clip8Display.ScaleFactor = 0.0017976811155676843
clip8Display.SelectScaleArray = 'WSSG'
clip8Display.GlyphType = 'Arrow'
clip8Display.GlyphTableIndexArray = 'WSSG'
clip8Display.DataAxesGrid = 'GridAxesRepresentation'
clip8Display.PolarAxes = 'PolarAxesRepresentation'
clip8Display.ScalarOpacityFunction = wSSGPWF
clip8Display.ScalarOpacityUnitDistance = 0.0005949075769969165
clip8Display.GaussianRadius = 0.0008988405577838422
clip8Display.SetScaleArray = ['POINTS', 'WSSG']
clip8Display.ScaleTransferFunction = 'PiecewiseFunction'
clip8Display.OpacityArray = ['POINTS', 'WSSG']
clip8Display.OpacityTransferFunction = 'PiecewiseFunction'

# hide data in view
Hide(clip7, renderView1)

# show color bar/color legend
clip8Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Clip'
clip9 = Clip(Input=clip8)
clip9.ClipType = 'Cylinder'
clip9.Scalars = ['POINTS', 'WSSG']
clip9.Value = 57862.67282051597

# init the 'Plane' selected for 'ClipType'
clip9.ClipType.Origin = clip4.ClipType.Origin
clip9.ClipType.Normal = clip4.ClipType.Normal

# set active source
SetActiveSource(clip4_1)

# set active source
SetActiveSource(clip9)

# show data in view
clip9Display = Show(clip9, renderView1)
# trace defaults for the display properties.
clip9Display.Representation = 'Surface'
clip9Display.ColorArrayName = ['POINTS', 'WSSG']
clip9Display.LookupTable = wSSGLUT
clip9Display.OSPRayScaleArray = 'WSSG'
clip9Display.OSPRayScaleFunction = 'PiecewiseFunction'
clip9Display.SelectOrientationVectors = 'velocity'
clip9Display.ScaleFactor = 0.0008415337651968002
clip9Display.SelectScaleArray = 'WSSG'
clip9Display.GlyphType = 'Arrow'
clip9Display.GlyphTableIndexArray = 'WSSG'
clip9Display.DataAxesGrid = 'GridAxesRepresentation'
clip9Display.PolarAxes = 'PolarAxesRepresentation'
clip9Display.ScalarOpacityFunction = wSSGPWF
clip9Display.ScalarOpacityUnitDistance = 0.0003871343211819996
clip9Display.GaussianRadius = 0.0004207668825984001
clip9Display.SetScaleArray = ['POINTS', 'WSSG']
clip9Display.ScaleTransferFunction = 'PiecewiseFunction'
clip9Display.OpacityArray = ['POINTS', 'WSSG']
clip9Display.OpacityTransferFunction = 'PiecewiseFunction'

# hide data in view
Hide(clip8, renderView1)

# show color bar/color legend
clip9Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Clip'
clip10 = Clip(Input=clip9)
clip10.ClipType = 'Plane'
clip10.Scalars = ['POINTS', 'WSSG']
clip10.Value = 57862.67282051597

# init the 'Plane' selected for 'ClipType'
clip10.ClipType.Origin = clip5.ClipType.Origin
clip10.ClipType.Normal = clip5.ClipType.Normal

# show data in view
clip10Display = Show(clip10, renderView1)
# trace defaults for the display properties.
clip10Display.Representation = 'Surface'
clip10Display.ColorArrayName = ['POINTS', 'WSSG']
clip10Display.LookupTable = wSSGLUT
clip10Display.OSPRayScaleArray = 'WSSG'
clip10Display.OSPRayScaleFunction = 'PiecewiseFunction'
clip10Display.SelectOrientationVectors = 'velocity'
clip10Display.ScaleFactor = 0.0008415337651968002
clip10Display.SelectScaleArray = 'WSSG'
clip10Display.GlyphType = 'Arrow'
clip10Display.GlyphTableIndexArray = 'WSSG'
clip10Display.DataAxesGrid = 'GridAxesRepresentation'
clip10Display.PolarAxes = 'PolarAxesRepresentation'
clip10Display.ScalarOpacityFunction = wSSGPWF
clip10Display.ScalarOpacityUnitDistance = 0.00040867737545182835
clip10Display.GaussianRadius = 0.0004207668825984001
clip10Display.SetScaleArray = ['POINTS', 'WSSG']
clip10Display.ScaleTransferFunction = 'PiecewiseFunction'
clip10Display.OpacityArray = ['POINTS', 'WSSG']
clip10Display.OpacityTransferFunction = 'PiecewiseFunction'

# hide data in view
Hide(clip9, renderView1)

# show color bar/color legend
clip10Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=clip10.ClipType)

# set scalar coloring
ColorBy(clip10Display, ('POINTS', 'WSS'))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(wSSGLUT, renderView1)

# rescale color and/or opacity maps used to include current data range
clip10Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
clip10Display.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'WSS'
wSSLUT = GetColorTransferFunction('WSS')

# hide color bar/color legend
clip10Display.SetScalarBarVisibility(renderView1, False)

# show color bar/color legend
clip10Display.SetScalarBarVisibility(renderView1, True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
wSSLUT.ApplyPreset('Preset', True)

# get opacity transfer function/opacity map for 'WSS'
wSSPWF = GetOpacityTransferFunction('WSS')

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
wSSPWF.ApplyPreset('Preset', True)


# Rescale transfer function
wSSPWF.RescaleTransferFunction(0.0, 20.0)


#### saving camera placements for all active views

# current camera placement for renderView1
renderView1.CameraPosition = [0.017793797850530816, 0.03844181492412608, -0.0116705739512379]
renderView1.CameraFocalPoint = [0.022224755957722702, 0.01388202607631686, -0.006600723718293029]
renderView1.CameraViewUp = [0.8541884706543088, 0.04721650696514745, -0.5178152740806929]
renderView1.CameraParallelScale = 0.006591100251657915


# find source
clip5 = FindSource('Clip5')

# create a new 'Slice'
slice1 = Slice(Input=clip5)
slice1.SliceType = 'Plane'
slice1.SliceOffsetValues = [0.0]

# init the 'Plane' selected for 'SliceType'
slice1.SliceType.Origin = [0.021204414014340613, 0.015371019007124493, -0.007136359101173561]
slice1.SliceType.Normal = [0.363350250097687, -0.9236488657417244, 0.12185798524501504]


# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
# renderView1.ViewSize = [1128, 649]

# get color transfer function/color map for 'OSI'
oSILUT = GetColorTransferFunction('OSI')

# show data in view
slice1Display = Show(slice1, renderView1)
# trace defaults for the display properties.
slice1Display.Representation = 'Surface'
slice1Display.ColorArrayName = ['POINTS', 'OSI']
slice1Display.LookupTable = oSILUT
slice1Display.OSPRayScaleArray = 'OSI'
slice1Display.OSPRayScaleFunction = 'PiecewiseFunction'
slice1Display.SelectOrientationVectors = 'velocity'
slice1Display.ScaleFactor = 0.0006727379513904452
slice1Display.SelectScaleArray = 'OSI'
slice1Display.GlyphType = 'Arrow'
slice1Display.GlyphTableIndexArray = 'OSI'
slice1Display.DataAxesGrid = 'GridAxesRepresentation'
slice1Display.PolarAxes = 'PolarAxesRepresentation'
slice1Display.GaussianRadius = 0.0003363689756952226
slice1Display.SetScaleArray = ['POINTS', 'OSI']
slice1Display.ScaleTransferFunction = 'PiecewiseFunction'
slice1Display.OpacityArray = ['POINTS', 'OSI']
slice1Display.OpacityTransferFunction = 'PiecewiseFunction'

# hide data in view
Hide(clip5, renderView1)

# show color bar/color legend
slice1Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Tube'
tube1 = Tube(Input=slice1)
tube1.Scalars = ['POINTS', 'OSI']
tube1.Vectors = ['POINTS', 'velocity']
tube1.Radius = 5e-05
tube1.NumberofSides = 8

# show data in view
tube1Display = Show(tube1, renderView1)
# trace defaults for the display properties.
tube1Display.Representation = 'Surface'
tube1Display.ColorArrayName = ['POINTS', 'OSI']
tube1Display.LookupTable = oSILUT
tube1Display.OSPRayScaleArray = 'OSI'
tube1Display.OSPRayScaleFunction = 'PiecewiseFunction'
tube1Display.SelectOrientationVectors = 'velocity'
tube1Display.ScaleFactor = 0.0006827377248555422
tube1Display.SelectScaleArray = 'OSI'
tube1Display.GlyphType = 'Arrow'
tube1Display.GlyphTableIndexArray = 'OSI'
tube1Display.DataAxesGrid = 'GridAxesRepresentation'
tube1Display.PolarAxes = 'PolarAxesRepresentation'
tube1Display.GaussianRadius = 0.0003413688624277711
tube1Display.SetScaleArray = ['POINTS', 'OSI']
tube1Display.ScaleTransferFunction = 'PiecewiseFunction'
tube1Display.OpacityArray = ['POINTS', 'OSI']
tube1Display.OpacityTransferFunction = 'PiecewiseFunction'

# hide data in view
Hide(slice1, renderView1)

# show color bar/color legend
tube1Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# set active source
SetActiveSource(slice1)

# reset view to fit data
renderView1.ResetCamera()

# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=slice1.SliceType)

# reset view to fit data
renderView1.ResetCamera()

# set active source
SetActiveSource(clip5)

# get opacity transfer function/opacity map for 'OSI'
oSIPWF = GetOpacityTransferFunction('OSI')

# show data in view
clip5Display = Show(clip5, renderView1)
# trace defaults for the display properties.
clip5Display.Representation = 'Surface'
clip5Display.ColorArrayName = ['POINTS', 'OSI']
clip5Display.LookupTable = oSILUT
clip5Display.OSPRayScaleArray = 'OSI'
clip5Display.OSPRayScaleFunction = 'PiecewiseFunction'
clip5Display.SelectOrientationVectors = 'velocity'
clip5Display.ScaleFactor = 0.00015334729105234148
clip5Display.SelectScaleArray = 'OSI'
clip5Display.GlyphType = 'Arrow'
clip5Display.GlyphTableIndexArray = 'OSI'
clip5Display.DataAxesGrid = 'GridAxesRepresentation'
clip5Display.PolarAxes = 'PolarAxesRepresentation'
clip5Display.ScalarOpacityFunction = oSIPWF
clip5Display.ScalarOpacityUnitDistance = 8.831366592932063e-05
clip5Display.GaussianRadius = 7.667364552617074e-05
clip5Display.SetScaleArray = ['POINTS', 'OSI']
clip5Display.ScaleTransferFunction = 'PiecewiseFunction'
clip5Display.OpacityArray = ['POINTS', 'OSI']
clip5Display.OpacityTransferFunction = 'PiecewiseFunction'

# show color bar/color legend
clip5Display.SetScalarBarVisibility(renderView1, True)

# hide data in view
Hide(clip5, renderView1)

# set active source
SetActiveSource(clip10)

# get color transfer function/color map for 'WSS'
wSSLUT = GetColorTransferFunction('WSS')

# create a new 'Slice'
slice2 = Slice(Input=clip10)
slice2.SliceType = 'Plane'
slice2.SliceOffsetValues = [0.0]

# init the 'Plane' selected for 'SliceType'
slice2.SliceType.Origin = slice1.SliceType.Origin
slice2.SliceType.Normal = slice1.SliceType.Normal

# set active source
SetActiveSource(slice1)

# set active source
SetActiveSource(slice2)

# get color transfer function/color map for 'WSSG'
wSSGLUT = GetColorTransferFunction('WSSG')

# show data in view
slice2Display = Show(slice2, renderView1)
# trace defaults for the display properties.
slice2Display.Representation = 'Surface'
slice2Display.ColorArrayName = ['POINTS', 'WSSG']
slice2Display.LookupTable = wSSGLUT
slice2Display.OSPRayScaleArray = 'WSSG'
slice2Display.OSPRayScaleFunction = 'PiecewiseFunction'
slice2Display.SelectOrientationVectors = 'velocity'
slice2Display.ScaleFactor = 0.0006727379513904452
slice2Display.SelectScaleArray = 'WSSG'
slice2Display.GlyphType = 'Arrow'
slice2Display.GlyphTableIndexArray = 'WSSG'
slice2Display.DataAxesGrid = 'GridAxesRepresentation'
slice2Display.PolarAxes = 'PolarAxesRepresentation'
slice2Display.GaussianRadius = 0.0003363689756952226
slice2Display.SetScaleArray = ['POINTS', 'WSSG']
slice2Display.ScaleTransferFunction = 'PiecewiseFunction'
slice2Display.OpacityArray = ['POINTS', 'WSSG']
slice2Display.OpacityTransferFunction = 'PiecewiseFunction'

# hide data in view
Hide(clip10, renderView1)

# show color bar/color legend
slice2Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# set active source
SetActiveSource(tube1)

# set active source
SetActiveSource(slice2)

# create a new 'Tube'
tube2 = Tube(Input=slice2)
tube2.Scalars = ['POINTS', 'WSSG']
tube2.Vectors = ['POINTS', 'velocity']
tube2.Radius = tube1.Radius
tube2.NumberofSides = tube1.NumberofSides

# set active source
SetActiveSource(tube1)

# set active source
SetActiveSource(tube2)

# Properties modified on tube2
tube2.Scalars = ['POINTS', 'WSS']

# show data in view
tube2Display = Show(tube2, renderView1)
# trace defaults for the display properties.
tube2Display.Representation = 'Surface'
tube2Display.ColorArrayName = ['POINTS', 'WSSG']
tube2Display.LookupTable = wSSGLUT
tube2Display.OSPRayScaleArray = 'WSSG'
tube2Display.OSPRayScaleFunction = 'PiecewiseFunction'
tube2Display.SelectOrientationVectors = 'velocity'
tube2Display.ScaleFactor = 0.0006827377248555422
tube2Display.SelectScaleArray = 'WSSG'
tube2Display.GlyphType = 'Arrow'
tube2Display.GlyphTableIndexArray = 'WSSG'
tube2Display.DataAxesGrid = 'GridAxesRepresentation'
tube2Display.PolarAxes = 'PolarAxesRepresentation'
tube2Display.GaussianRadius = 0.0003413688624277711
tube2Display.SetScaleArray = ['POINTS', 'WSSG']
tube2Display.ScaleTransferFunction = 'PiecewiseFunction'
tube2Display.OpacityArray = ['POINTS', 'WSSG']
tube2Display.OpacityTransferFunction = 'PiecewiseFunction'

# hide data in view
Hide(slice2, renderView1)

# show color bar/color legend
tube2Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# hide data in view
Hide(tube1, renderView1)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
wSSGLUT.ApplyPreset('Preset', True)

# get opacity transfer function/opacity map for 'WSSG'
wSSGPWF = GetOpacityTransferFunction('WSSG')

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
wSSGPWF.ApplyPreset('Preset', True)

# set scalar coloring
ColorBy(tube2Display, ('POINTS', 'WSS'))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(wSSGLUT, renderView1)

# rescale color and/or opacity maps used to include current data range
tube2Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
tube2Display.SetScalarBarVisibility(renderView1, True)

# set scalar coloring
ColorBy(tube2Display, ('POINTS', 'TubeNormals', 'Magnitude'))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(wSSLUT, renderView1)

# rescale color and/or opacity maps used to include current data range
tube2Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
tube2Display.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'TubeNormals'
tubeNormalsLUT = GetColorTransferFunction('TubeNormals')

# set scalar coloring
ColorBy(tube2Display, ('POINTS', 'WSS'))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(tubeNormalsLUT, renderView1)

# rescale color and/or opacity maps used to include current data range
tube2Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
tube2Display.SetScalarBarVisibility(renderView1, True)



#### uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).
