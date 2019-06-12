#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'XML Unstructured Grid Reader'
xMLUnstructuredGridReader1 = XMLUnstructuredGridReader(FileName=['/raid/home/ksansom/caseFiles/mri/VWI_proj/case4/fluent_dsa/vtk_out/calc_test_node_stats.vtu'])
xMLUnstructuredGridReader1.PointArrayStatus = ['WSS', 'WSSG', 'absolute_pressure', 'TAWSS', 'TAWSSG', 'OSI', 'velocity']

# create a new 'XML Unstructured Grid Reader'
xMLUnstructuredGridReader2 = XMLUnstructuredGridReader(FileName=['/raid/home/ksansom/caseFiles/mri/VWI_proj/case4/fluent_dsa/vtk_out/calc_test_node.vtu'])
xMLUnstructuredGridReader2.PointArrayStatus = ['absolute_pressure', 'velocity', 'x_velocity', 'x_wall_shear', 'y_velocity', 'y_wall_shear', 'z_velocity', 'z_wall_shear', 'WSS', 'x_WSS_grad', 'y_WSS_grad', 'z_WSS_grad', 'WSSG']

# get animation scene
animationScene1 = GetAnimationScene()

# update animation scene based on data timesteps
animationScene1.UpdateAnimationUsingDataTimeSteps()

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
# renderView1.ViewSize = [928, 595]

# reset view to fit data
renderView1.ResetCamera()

# get color transfer function/color map for 'OSI'
oSILUT = GetColorTransferFunction('OSI')

# get opacity transfer function/opacity map for 'OSI'
oSIPWF = GetOpacityTransferFunction('OSI')

# show data in view
xMLUnstructuredGridReader1Display = Show(xMLUnstructuredGridReader1, renderView1)
# trace defaults for the display properties.
xMLUnstructuredGridReader1Display.Representation = 'Surface'
xMLUnstructuredGridReader1Display.ColorArrayName = ['POINTS', 'OSI']
xMLUnstructuredGridReader1Display.LookupTable = oSILUT
xMLUnstructuredGridReader1Display.OSPRayScaleArray = 'OSI'
xMLUnstructuredGridReader1Display.OSPRayScaleFunction = 'PiecewiseFunction'
xMLUnstructuredGridReader1Display.SelectOrientationVectors = 'velocity'
xMLUnstructuredGridReader1Display.ScaleFactor = 0.002641490660607815
xMLUnstructuredGridReader1Display.SelectScaleArray = 'OSI'
xMLUnstructuredGridReader1Display.GlyphType = 'Arrow'
xMLUnstructuredGridReader1Display.GlyphTableIndexArray = 'OSI'
xMLUnstructuredGridReader1Display.DataAxesGrid = 'GridAxesRepresentation'
xMLUnstructuredGridReader1Display.PolarAxes = 'PolarAxesRepresentation'
xMLUnstructuredGridReader1Display.ScalarOpacityFunction = oSIPWF
xMLUnstructuredGridReader1Display.ScalarOpacityUnitDistance = 0.0006752548306746431
xMLUnstructuredGridReader1Display.GaussianRadius = 0.0013207453303039074
xMLUnstructuredGridReader1Display.SetScaleArray = ['POINTS', 'OSI']
xMLUnstructuredGridReader1Display.ScaleTransferFunction = 'PiecewiseFunction'
xMLUnstructuredGridReader1Display.OpacityArray = ['POINTS', 'OSI']
xMLUnstructuredGridReader1Display.OpacityTransferFunction = 'PiecewiseFunction'

# reset view to fit data
renderView1.ResetCamera()

# show color bar/color legend
xMLUnstructuredGridReader1Display.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'WSSG'
wSSGLUT = GetColorTransferFunction('WSSG')

# get opacity transfer function/opacity map for 'WSSG'
wSSGPWF = GetOpacityTransferFunction('WSSG')

# show data in view
xMLUnstructuredGridReader2Display = Show(xMLUnstructuredGridReader2, renderView1)
# trace defaults for the display properties.
xMLUnstructuredGridReader2Display.Representation = 'Surface'
xMLUnstructuredGridReader2Display.ColorArrayName = ['POINTS', 'WSSG']
xMLUnstructuredGridReader2Display.LookupTable = wSSGLUT
xMLUnstructuredGridReader2Display.OSPRayScaleArray = 'WSSG'
xMLUnstructuredGridReader2Display.OSPRayScaleFunction = 'PiecewiseFunction'
xMLUnstructuredGridReader2Display.SelectOrientationVectors = 'velocity'
xMLUnstructuredGridReader2Display.ScaleFactor = 0.002641490660607815
xMLUnstructuredGridReader2Display.SelectScaleArray = 'WSSG'
xMLUnstructuredGridReader2Display.GlyphType = 'Arrow'
xMLUnstructuredGridReader2Display.GlyphTableIndexArray = 'WSSG'
xMLUnstructuredGridReader2Display.DataAxesGrid = 'GridAxesRepresentation'
xMLUnstructuredGridReader2Display.PolarAxes = 'PolarAxesRepresentation'
xMLUnstructuredGridReader2Display.ScalarOpacityFunction = wSSGPWF
xMLUnstructuredGridReader2Display.ScalarOpacityUnitDistance = 0.0006752548306746431
xMLUnstructuredGridReader2Display.GaussianRadius = 0.0013207453303039074
xMLUnstructuredGridReader2Display.SetScaleArray = ['POINTS', 'WSSG']
xMLUnstructuredGridReader2Display.ScaleTransferFunction = 'PiecewiseFunction'
xMLUnstructuredGridReader2Display.OpacityArray = ['POINTS', 'WSSG']
xMLUnstructuredGridReader2Display.OpacityTransferFunction = 'PiecewiseFunction'

# show color bar/color legend
xMLUnstructuredGridReader2Display.SetScalarBarVisibility(renderView1, True)

# show data in view
xMLUnstructuredGridReader2Display = Show(xMLUnstructuredGridReader2, renderView1)

# hide color bar/color legend
xMLUnstructuredGridReader2Display.SetScalarBarVisibility(renderView1, False)

# hide data in view
Hide(xMLUnstructuredGridReader1, renderView1)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
wSSGLUT.ApplyPreset('Preset', True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
wSSGPWF.ApplyPreset('Preset', True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
wSSGLUT.ApplyPreset('Preset', True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
wSSGPWF.ApplyPreset('Preset', True)

# set active source
SetActiveSource(xMLUnstructuredGridReader1)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
oSILUT.ApplyPreset('Preset', True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
oSIPWF.ApplyPreset('Preset', True)

# create a new 'Extract Surface'
extractSurface1 = ExtractSurface(Input=xMLUnstructuredGridReader1)

# show data in view
extractSurface1Display = Show(extractSurface1, renderView1)
# trace defaults for the display properties.
extractSurface1Display.Representation = 'Surface'
extractSurface1Display.ColorArrayName = ['POINTS', 'OSI']
extractSurface1Display.LookupTable = oSILUT
extractSurface1Display.OSPRayScaleArray = 'OSI'
extractSurface1Display.OSPRayScaleFunction = 'PiecewiseFunction'
extractSurface1Display.SelectOrientationVectors = 'velocity'
extractSurface1Display.ScaleFactor = 0.002641490660607815
extractSurface1Display.SelectScaleArray = 'OSI'
extractSurface1Display.GlyphType = 'Arrow'
extractSurface1Display.GlyphTableIndexArray = 'OSI'
extractSurface1Display.DataAxesGrid = 'GridAxesRepresentation'
extractSurface1Display.PolarAxes = 'PolarAxesRepresentation'
extractSurface1Display.GaussianRadius = 0.0013207453303039074
extractSurface1Display.SetScaleArray = ['POINTS', 'OSI']
extractSurface1Display.ScaleTransferFunction = 'PiecewiseFunction'
extractSurface1Display.OpacityArray = ['POINTS', 'OSI']
extractSurface1Display.OpacityTransferFunction = 'PiecewiseFunction'

# hide data in view
Hide(xMLUnstructuredGridReader1, renderView1)

# hide color bar/color legend
extractSurface1Display.SetScalarBarVisibility(renderView1, False)

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(oSILUT, renderView1)

# hide data in view
Hide(extractSurface1, renderView1)

# hide data in view
Hide(xMLUnstructuredGridReader1, renderView1)

# hide data in view
Hide(xMLUnstructuredGridReader2, renderView1)

# create a new 'Clip'
clip1 = Clip(Input=xMLUnstructuredGridReader1)
clip1.ClipType = 'Plane'
clip1.Scalars = ['POINTS', 'OSI']
clip1.Value = -8.963760541398452

# init the 'Plane' selected for 'ClipType'
clip1.ClipType.Origin = [0.019185381077859477, 0.008678421124464417, 0.019721650218845817]
clip1.ClipType.Normal = [-0.9140283519495734, -0.03315688808077104, 0.40429295393952386]
# toggle 3D widget visibility (only when running from the GUI)
Show3DWidgets(proxy=clip1.ClipType)

# show data in view
clip1Display = Show(clip1, renderView1)
# trace defaults for the display properties.
clip1Display.Representation = 'Surface'
clip1Display.ColorArrayName = ['POINTS', 'OSI']
clip1Display.LookupTable = oSILUT
clip1Display.OSPRayScaleArray = 'OSI'
clip1Display.OSPRayScaleFunction = 'PiecewiseFunction'
clip1Display.SelectOrientationVectors = 'velocity'
clip1Display.ScaleFactor = 0.0012495411559939385
clip1Display.SelectScaleArray = 'OSI'
clip1Display.GlyphType = 'Arrow'
clip1Display.GlyphTableIndexArray = 'OSI'
clip1Display.DataAxesGrid = 'GridAxesRepresentation'
clip1Display.PolarAxes = 'PolarAxesRepresentation'
clip1Display.ScalarOpacityFunction = oSIPWF
clip1Display.ScalarOpacityUnitDistance = 0.0004738466035644152
clip1Display.GaussianRadius = 0.0006247705779969693
clip1Display.SetScaleArray = ['POINTS', 'OSI']
clip1Display.ScaleTransferFunction = 'PiecewiseFunction'
clip1Display.OpacityArray = ['POINTS', 'OSI']
clip1Display.OpacityTransferFunction = 'PiecewiseFunction'

# hide color bar/color legend
clip1Display.SetScalarBarVisibility(renderView1, False)

# hide data in view
Hide(clip1, renderView1)

# create a new 'Clip'
clip2 = Clip(Input=clip1)
clip2.ClipType = 'Plane'
clip2.Scalars = ['POINTS', 'OSI']
clip2.Value = -4.334532492423367

# init the 'Plane' selected for 'ClipType'
clip2.ClipType.Origin = [0.016463510161933183, 0.013672129568797606, 0.020951286955119792]
clip2.ClipType.Normal = [-0.14808092203454595, -0.9422076182376612, -0.30052761048581295]

# toggle 3D widget visibility (only when running from the GUI)
Show3DWidgets(proxy=clip2.ClipType)

# show data in view
clip2Display = Show(clip2, renderView1)
# trace defaults for the display properties.
clip2Display.Representation = 'Surface'
clip2Display.ColorArrayName = ['POINTS', 'OSI']
clip2Display.LookupTable = oSILUT
clip2Display.OSPRayScaleArray = 'OSI'
clip2Display.OSPRayScaleFunction = 'PiecewiseFunction'
clip2Display.SelectOrientationVectors = 'velocity'
clip2Display.ScaleFactor = 0.0012495411559939385
clip2Display.SelectScaleArray = 'OSI'
clip2Display.GlyphType = 'Arrow'
clip2Display.GlyphTableIndexArray = 'OSI'
clip2Display.DataAxesGrid = 'GridAxesRepresentation'
clip2Display.PolarAxes = 'PolarAxesRepresentation'
clip2Display.ScalarOpacityFunction = oSIPWF
clip2Display.ScalarOpacityUnitDistance = 0.0004570996202003559
clip2Display.GaussianRadius = 0.0006247705779969693
clip2Display.SetScaleArray = ['POINTS', 'OSI']
clip2Display.ScaleTransferFunction = 'PiecewiseFunction'
clip2Display.OpacityArray = ['POINTS', 'OSI']
clip2Display.OpacityTransferFunction = 'PiecewiseFunction'

# hide color bar/color legend
clip2Display.SetScalarBarVisibility(renderView1, False)

# hide data in view
Hide(clip2, renderView1)

# create a new 'Clip'
clip3 = Clip(Input=clip2)
clip3.ClipType = 'Plane'
clip3.Scalars = ['POINTS', 'OSI']
clip3.Value = -4.334532492423367

# init the 'Plane' selected for 'ClipType'
clip3.ClipType.Origin = [0.016711021877632924, 0.009538794369297741, 0.02097712830393285]
clip3.ClipType.Normal = [-0.2022641960827218, 0.8606682429625938, -0.46726798578405937]

# toggle 3D widget visibility (only when running from the GUI)
Show3DWidgets(proxy=clip3.ClipType)

# show data in view
clip3Display = Show(clip3, renderView1)
# trace defaults for the display properties.
clip3Display.Representation = 'Surface'
clip3Display.ColorArrayName = ['POINTS', 'OSI']
clip3Display.LookupTable = oSILUT
clip3Display.OSPRayScaleArray = 'OSI'
clip3Display.OSPRayScaleFunction = 'PiecewiseFunction'
clip3Display.SelectOrientationVectors = 'velocity'
clip3Display.ScaleFactor = 0.0012495411559939385
clip3Display.SelectScaleArray = 'OSI'
clip3Display.GlyphType = 'Arrow'
clip3Display.GlyphTableIndexArray = 'OSI'
clip3Display.DataAxesGrid = 'GridAxesRepresentation'
clip3Display.PolarAxes = 'PolarAxesRepresentation'
clip3Display.ScalarOpacityFunction = oSIPWF
clip3Display.ScalarOpacityUnitDistance = 0.0004945866681605259
clip3Display.GaussianRadius = 0.0006247705779969693
clip3Display.SetScaleArray = ['POINTS', 'OSI']
clip3Display.ScaleTransferFunction = 'PiecewiseFunction'
clip3Display.OpacityArray = ['POINTS', 'OSI']
clip3Display.OpacityTransferFunction = 'PiecewiseFunction'

# hide color bar/color legend
clip3Display.SetScalarBarVisibility(renderView1, False)

# hide data in view
Hide(clip3, renderView1)

# create a new 'Clip'
clip4 = Clip(Input=clip3)
clip4.ClipType = 'Plane'
clip4.Scalars = ['POINTS', 'OSI']
clip4.Value = -0.22316894760113207

# init the 'Plane' selected for 'ClipType'
clip4.ClipType.Origin = [0.013133594821495129, 0.009304044349573848, 0.020326088763650343]
clip4.ClipType.Normal = [0.5663667872989001, 0.8238781757097235, 0.02129351624181657]

# toggle 3D widget visibility (only when running from the GUI)
Show3DWidgets(proxy=clip4.ClipType)

# show data in view
clip4Display = Show(clip4, renderView1)
# trace defaults for the display properties.
clip4Display.Representation = 'Surface'
clip4Display.ColorArrayName = ['POINTS', 'OSI']
clip4Display.LookupTable = oSILUT
clip4Display.OSPRayScaleArray = 'OSI'
clip4Display.OSPRayScaleFunction = 'PiecewiseFunction'
clip4Display.SelectOrientationVectors = 'velocity'
clip4Display.ScaleFactor = 0.0007063730619847775
clip4Display.SelectScaleArray = 'OSI'
clip4Display.GlyphType = 'Arrow'
clip4Display.GlyphTableIndexArray = 'OSI'
clip4Display.DataAxesGrid = 'GridAxesRepresentation'
clip4Display.PolarAxes = 'PolarAxesRepresentation'
clip4Display.ScalarOpacityFunction = oSIPWF
clip4Display.ScalarOpacityUnitDistance = 0.00037060974906745676
clip4Display.GaussianRadius = 0.00035318653099238875
clip4Display.SetScaleArray = ['POINTS', 'OSI']
clip4Display.ScaleTransferFunction = 'PiecewiseFunction'
clip4Display.OpacityArray = ['POINTS', 'OSI']
clip4Display.OpacityTransferFunction = 'PiecewiseFunction'

# hide color bar/color legend
clip4Display.SetScalarBarVisibility(renderView1, False)

# hide data in view
Hide(clip4, renderView1)

# create a new 'Clip'
clip5 = Clip(Input=clip4)
clip5.ClipType = 'Cylinder'
clip5.Scalars = ['POINTS', 'OSI']
clip5.Value = -0.22316894760113207

# init the 'Plane' selected for 'ClipType'
clip5.ClipType.Center = [0.018777387893322797, 0.01057777265095785, 0.021172440223998248]
clip5.ClipType.Axis = [0.36272981418484473, 0.7899369662882306, -0.4943952580605674]
clip5.ClipType.Radius = 0.0021806721886416867

# toggle 3D widget visibility (only when running from the GUI)
Show3DWidgets(proxy=clip5.ClipType)

# show data in view
clip5Display = Show(clip5, renderView1)
# trace defaults for the display properties.
clip5Display.Representation = 'Surface'
clip5Display.ColorArrayName = ['POINTS', 'OSI']
clip5Display.LookupTable = oSILUT
clip5Display.OSPRayScaleArray = 'OSI'
clip5Display.OSPRayScaleFunction = 'PiecewiseFunction'
clip5Display.SelectOrientationVectors = 'velocity'
clip5Display.ScaleFactor = 0.0005240732803940773
clip5Display.SelectScaleArray = 'OSI'
clip5Display.GlyphType = 'Arrow'
clip5Display.GlyphTableIndexArray = 'OSI'
clip5Display.DataAxesGrid = 'GridAxesRepresentation'
clip5Display.PolarAxes = 'PolarAxesRepresentation'
clip5Display.ScalarOpacityFunction = oSIPWF
clip5Display.ScalarOpacityUnitDistance = 0.0003557483835107667
clip5Display.GaussianRadius = 0.00026203664019703866
clip5Display.SetScaleArray = ['POINTS', 'OSI']
clip5Display.ScaleTransferFunction = 'PiecewiseFunction'
clip5Display.OpacityArray = ['POINTS', 'OSI']
clip5Display.OpacityTransferFunction = 'PiecewiseFunction'

# hide color bar/color legend
clip5Display.SetScalarBarVisibility(renderView1, False)

# Rescale transfer function
oSILUT.RescaleTransferFunction(0.0, 0.5)

# Rescale transfer function
oSIPWF.RescaleTransferFunction(0.0, 0.5)

# hide data in view
Hide(clip5, renderView1)

# create a new 'Slice'
slice1 = Slice(Input=clip5)
slice1.SliceType = 'Plane'
slice1.SliceOffsetValues = [0.0]

# init the 'Plane' selected for 'SliceType'
slice1.SliceType.Origin = [0.01406339196240306, 0.011922889508895057, 0.02007914092262216]
slice1.SliceType.Normal = [0.9166910278672833, -0.3166687289041705, 0.2437180246962479]

# toggle 3D widget visibility (only when running from the GUI)
Show3DWidgets(proxy=slice1.SliceType)

# show data in view
slice1Display = Show(slice1, renderView1)
# trace defaults for the display properties.
slice1Display.Representation = 'Surface'
slice1Display.ColorArrayName = ['POINTS', 'OSI']
slice1Display.LookupTable = oSILUT
slice1Display.OSPRayScaleArray = 'OSI'
slice1Display.OSPRayScaleFunction = 'PiecewiseFunction'
slice1Display.SelectOrientationVectors = 'velocity'
slice1Display.ScaleFactor = 0.00044556772336363794
slice1Display.SelectScaleArray = 'OSI'
slice1Display.GlyphType = 'Arrow'
slice1Display.GlyphTableIndexArray = 'OSI'
slice1Display.DataAxesGrid = 'GridAxesRepresentation'
slice1Display.PolarAxes = 'PolarAxesRepresentation'
slice1Display.GaussianRadius = 0.00022278386168181897
slice1Display.SetScaleArray = ['POINTS', 'OSI']
slice1Display.ScaleTransferFunction = 'PiecewiseFunction'
slice1Display.OpacityArray = ['POINTS', 'OSI']
slice1Display.OpacityTransferFunction = 'PiecewiseFunction'

# hide color bar/color legend
slice1Display.SetScalarBarVisibility(renderView1, False)

# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=slice1.SliceType)

# hide data in view
Hide(slice1, renderView1)

# create a new 'Tube'
tube1 = Tube(Input=slice1)
tube1.Scalars = ['POINTS', 'OSI']
tube1.Vectors = ['POINTS', 'velocity']
tube1.Radius = 5.0e-5
tube1.Capping = 0
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
tube1Display.ScaleFactor = 0.0004555660299956799
tube1Display.SelectScaleArray = 'OSI'
tube1Display.GlyphType = 'Arrow'
tube1Display.GlyphTableIndexArray = 'OSI'
tube1Display.DataAxesGrid = 'GridAxesRepresentation'
tube1Display.PolarAxes = 'PolarAxesRepresentation'
tube1Display.GaussianRadius = 0.00022778301499783995
tube1Display.SetScaleArray = ['POINTS', 'OSI']
tube1Display.ScaleTransferFunction = 'PiecewiseFunction'
tube1Display.OpacityArray = ['POINTS', 'OSI']
tube1Display.OpacityTransferFunction = 'PiecewiseFunction'

# hide color bar/color legend
tube1Display.SetScalarBarVisibility(renderView1, False)

# set active source
SetActiveSource(slice1)

# reset view to fit data
renderView1.ResetCamera()

# create a new 'Clip'
clip6 = Clip(Input=xMLUnstructuredGridReader2)
clip6.ClipType = 'Plane'
clip6.Scalars = ['POINTS', 'OSI']

# init the 'Plane' selected for 'ClipType'
clip6.ClipType.Origin = clip1.ClipType.Origin
clip6.ClipType.Normal = clip1.ClipType.Normal

# toggle 3D widget visibility (only when running from the GUI)
Show3DWidgets(proxy=clip6.ClipType)

# show data in view
clip6Display = Show(clip6, renderView1)
# trace defaults for the display properties.
clip6Display.Representation = 'Surface'
clip6Display.ColorArrayName = ['POINTS', 'WSSG']
clip6Display.LookupTable = wSSGLUT
clip6Display.OSPRayScaleArray = 'WSSG'
clip6Display.OSPRayScaleFunction = 'PiecewiseFunction'
clip6Display.SelectOrientationVectors = 'velocity'
clip6Display.ScaleFactor = 0.0012495411559939385
clip6Display.SelectScaleArray = 'WSSG'
clip6Display.GlyphType = 'Arrow'
clip6Display.GlyphTableIndexArray = 'WSSG'
clip6Display.DataAxesGrid = 'GridAxesRepresentation'
clip6Display.PolarAxes = 'PolarAxesRepresentation'
clip6Display.ScalarOpacityFunction = wSSGPWF
clip6Display.ScalarOpacityUnitDistance = 0.0004738466035644152
clip6Display.GaussianRadius = 0.0006247705779969693
clip6Display.SetScaleArray = ['POINTS', 'WSSG']
clip6Display.ScaleTransferFunction = 'PiecewiseFunction'
clip6Display.OpacityArray = ['POINTS', 'WSSG']
clip6Display.OpacityTransferFunction = 'PiecewiseFunction'

# hide color bar/color legend
clip6Display.SetScalarBarVisibility(renderView1, False)

# hide data in view
Hide(clip6, renderView1)

# create a new 'Clip'
clip7 = Clip(Input=clip6)
clip7.ClipType = 'Plane'
clip7.Scalars = ['POINTS', 'WSSG']
clip7.Value = clip3.Value

# init the 'Plane' selected for 'ClipType'
clip7.ClipType.Origin = clip2.ClipType.Origin
clip7.ClipType.Normal = clip2.ClipType.Normal
# toggle 3D widget visibility (only when running from the GUI)
Show3DWidgets(proxy=clip7.ClipType)

# show data in view
clip7Display = Show(clip7, renderView1)
# trace defaults for the display properties.
clip7Display.Representation = 'Surface'
clip7Display.ColorArrayName = ['POINTS', 'WSSG']
clip7Display.LookupTable = wSSGLUT
clip7Display.OSPRayScaleArray = 'WSSG'
clip7Display.OSPRayScaleFunction = 'PiecewiseFunction'
clip7Display.SelectOrientationVectors = 'velocity'
clip7Display.ScaleFactor = 0.0012495411559939385
clip7Display.SelectScaleArray = 'WSSG'
clip7Display.GlyphType = 'Arrow'
clip7Display.GlyphTableIndexArray = 'WSSG'
clip7Display.DataAxesGrid = 'GridAxesRepresentation'
clip7Display.PolarAxes = 'PolarAxesRepresentation'
clip7Display.ScalarOpacityFunction = wSSGPWF
clip7Display.ScalarOpacityUnitDistance = 0.0004570996202003559
clip7Display.GaussianRadius = 0.0006247705779969693
clip7Display.SetScaleArray = ['POINTS', 'WSSG']
clip7Display.ScaleTransferFunction = 'PiecewiseFunction'
clip7Display.OpacityArray = ['POINTS', 'WSSG']
clip7Display.OpacityTransferFunction = 'PiecewiseFunction'

# hide color bar/color legend
clip7Display.SetScalarBarVisibility(renderView1, False)

# hide data in view
Hide(clip7, renderView1)

# create a new 'Clip'
clip8 = Clip(Input=clip7)
clip8.ClipType = 'Plane'
clip8.Scalars = ['POINTS', 'WSSG']
clip8.Value = clip4.Value

# init the 'Plane' selected for 'ClipType'
clip8.ClipType.Origin = clip3.ClipType.Origin
clip8.ClipType.Normal = clip3.ClipType.Normal
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
clip8Display.ScaleFactor = 0.0012495411559939385
clip8Display.SelectScaleArray = 'WSSG'
clip8Display.GlyphType = 'Arrow'
clip8Display.GlyphTableIndexArray = 'WSSG'
clip8Display.DataAxesGrid = 'GridAxesRepresentation'
clip8Display.PolarAxes = 'PolarAxesRepresentation'
clip8Display.ScalarOpacityFunction = wSSGPWF
clip8Display.ScalarOpacityUnitDistance = 0.0004945866681605259
clip8Display.GaussianRadius = 0.0006247705779969693
clip8Display.SetScaleArray = ['POINTS', 'WSSG']
clip8Display.ScaleTransferFunction = 'PiecewiseFunction'
clip8Display.OpacityArray = ['POINTS', 'WSSG']
clip8Display.OpacityTransferFunction = 'PiecewiseFunction'

# hide color bar/color legend
clip8Display.SetScalarBarVisibility(renderView1, False)

# hide data in view
Hide(clip8, renderView1)

# create a new 'Clip'
clip9 = Clip(Input=clip8)
clip9.ClipType = 'Plane'
clip9.Scalars = ['POINTS', 'WSSG']
clip9.Value = 12858.09077513687

# init the 'Plane' selected for 'ClipType'
clip9.ClipType.Origin = clip4.ClipType.Origin
clip9.ClipType.Normal = clip4.ClipType.Normal
# toggle 3D widget visibility (only when running from the GUI)
Show3DWidgets(proxy=clip9.ClipType)

# show data in view
clip9Display = Show(clip9, renderView1)
# trace defaults for the display properties.
clip9Display.Representation = 'Surface'
clip9Display.ColorArrayName = ['POINTS', 'WSSG']
clip9Display.LookupTable = wSSGLUT
clip9Display.OSPRayScaleArray = 'WSSG'
clip9Display.OSPRayScaleFunction = 'PiecewiseFunction'
clip9Display.SelectOrientationVectors = 'velocity'
clip9Display.ScaleFactor = 0.0007063730619847775
clip9Display.SelectScaleArray = 'WSSG'
clip9Display.GlyphType = 'Arrow'
clip9Display.GlyphTableIndexArray = 'WSSG'
clip9Display.DataAxesGrid = 'GridAxesRepresentation'
clip9Display.PolarAxes = 'PolarAxesRepresentation'
clip9Display.ScalarOpacityFunction = wSSGPWF
clip9Display.ScalarOpacityUnitDistance = 0.00037060974906745676
clip9Display.GaussianRadius = 0.00035318653099238875
clip9Display.SetScaleArray = ['POINTS', 'WSSG']
clip9Display.ScaleTransferFunction = 'PiecewiseFunction'
clip9Display.OpacityArray = ['POINTS', 'WSSG']
clip9Display.OpacityTransferFunction = 'PiecewiseFunction'

# hide color bar/color legend
clip9Display.SetScalarBarVisibility(renderView1, False)

# hide data in view
Hide(clip9, renderView1)

# create a new 'Clip'
clip10 = Clip(Input=clip9)
clip10.ClipType = 'Cylinder'
clip10.Scalars = ['POINTS', 'WSSG']
clip10.Value = clip5.Value

# init the 'Plane' selected for 'ClipType'
clip10.ClipType.Center = clip5.ClipType.Center
clip10.ClipType.Axis = clip5.ClipType.Axis
clip10.ClipType.Radius = clip5.ClipType.Radius
# toggle 3D widget visibility (only when running from the GUI)
Show3DWidgets(proxy=clip10.ClipType)

# show data in view
clip10Display = Show(clip10, renderView1)
# trace defaults for the display properties.
clip10Display.Representation = 'Surface'
clip10Display.ColorArrayName = ['POINTS', 'WSSG']
clip10Display.LookupTable = wSSGLUT
clip10Display.OSPRayScaleArray = 'WSSG'
clip10Display.OSPRayScaleFunction = 'PiecewiseFunction'
clip10Display.SelectOrientationVectors = 'velocity'
clip10Display.ScaleFactor = 0.0005240732803940773
clip10Display.SelectScaleArray = 'WSSG'
clip10Display.GlyphType = 'Arrow'
clip10Display.GlyphTableIndexArray = 'WSSG'
clip10Display.DataAxesGrid = 'GridAxesRepresentation'
clip10Display.PolarAxes = 'PolarAxesRepresentation'
clip10Display.ScalarOpacityFunction = wSSGPWF
clip10Display.ScalarOpacityUnitDistance = 0.0003557483835107667
clip10Display.GaussianRadius = 0.00026203664019703866
clip10Display.SetScaleArray = ['POINTS', 'WSSG']
clip10Display.ScaleTransferFunction = 'PiecewiseFunction'
clip10Display.OpacityArray = ['POINTS', 'WSSG']
clip10Display.OpacityTransferFunction = 'PiecewiseFunction'

# hide color bar/color legend
clip10Display.SetScalarBarVisibility(renderView1, False)

# Rescale transfer function
oSILUT.RescaleTransferFunction(0.0, 0.5)

# Rescale transfer function
oSIPWF.RescaleTransferFunction(0.0, 0.5)

# hide data in view
Hide(clip10, renderView1)

# create a new 'Slice'
slice2 = Slice(Input=clip10)
slice2.SliceType = 'Plane'
slice2.SliceOffsetValues = [0.0]

# init the 'Plane' selected for 'SliceType'
slice2.SliceType.Origin = slice1.SliceType.Origin
slice2.SliceType.Normal = slice1.SliceType.Normal

# toggle 3D widget visibility (only when running from the GUI)
Show3DWidgets(proxy=slice2.SliceType)

# show data in view
slice2Display = Show(slice2, renderView1)
# trace defaults for the display properties.
slice2Display.Representation = 'Surface'
slice2Display.ColorArrayName = ['POINTS', 'WSSG']
slice2Display.LookupTable = wSSGLUT
slice2Display.OSPRayScaleArray = 'WSSG'
slice2Display.OSPRayScaleFunction = 'PiecewiseFunction'
slice2Display.SelectOrientationVectors = 'velocity'
slice2Display.ScaleFactor = 0.00044556772336363794
slice2Display.SelectScaleArray = 'WSSG'
slice2Display.GlyphType = 'Arrow'
slice2Display.GlyphTableIndexArray = 'WSSG'
slice2Display.DataAxesGrid = 'GridAxesRepresentation'
slice2Display.PolarAxes = 'PolarAxesRepresentation'
slice2Display.GaussianRadius = 0.00022278386168181897
slice2Display.SetScaleArray = ['POINTS', 'WSSG']
slice2Display.ScaleTransferFunction = 'PiecewiseFunction'
slice2Display.OpacityArray = ['POINTS', 'WSSG']
slice2Display.OpacityTransferFunction = 'PiecewiseFunction'

# hide color bar/color legend
slice2Display.SetScalarBarVisibility(renderView1, False)

# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=slice2.SliceType)

# hide data in view
Hide(slice2, renderView1)

# create a new 'Tube'
tube2 = Tube(Input=slice2)
tube2.Scalars = ['POINTS', 'WSSG']
tube2.Vectors = ['POINTS', 'velocity']
tube2.Radius = 4.4556772336363794e-05

# show data in view
tube2Display = Show(tube2, renderView1)
# trace defaults for the display properties.
tube2Display.Representation = 'Surface'
tube2Display.ColorArrayName = ['POINTS', 'WSSG']
tube2Display.LookupTable = wSSGLUT
tube2Display.OSPRayScaleArray = 'WSSG'
tube2Display.OSPRayScaleFunction = 'PiecewiseFunction'
tube2Display.SelectOrientationVectors = 'velocity'
tube2Display.ScaleFactor = 0.0004555660299956799
tube2Display.SelectScaleArray = 'WSSG'
tube2Display.GlyphType = 'Arrow'
tube2Display.GlyphTableIndexArray = 'WSSG'
tube2Display.DataAxesGrid = 'GridAxesRepresentation'
tube2Display.PolarAxes = 'PolarAxesRepresentation'
tube2Display.GaussianRadius = 0.00022778301499783995
tube2Display.SetScaleArray = ['POINTS', 'WSSG']
tube2Display.ScaleTransferFunction = 'PiecewiseFunction'
tube2Display.OpacityArray = ['POINTS', 'WSSG']
tube2Display.OpacityTransferFunction = 'PiecewiseFunction'

# hide color bar/color legend
tube2Display.SetScalarBarVisibility(renderView1, False)

# set active source
SetActiveSource(slice2)

# reset view to fit data
renderView1.ResetCamera()

# set active source
SetActiveSource(xMLUnstructuredGridReader1)

# show data in view
xMLUnstructuredGridReader1Display = Show(xMLUnstructuredGridReader1, renderView1)

# show color bar/color legend
xMLUnstructuredGridReader1Display.SetScalarBarVisibility(renderView1, True)

# reset view to fit data
renderView1.ResetCamera()

ReloadFiles(xMLUnstructuredGridReader1)

# hide data in view
Hide(xMLUnstructuredGridReader1, renderView1)

#### saving camera placements for all active views

# current camera placement for renderView1
renderView1.CameraPosition = [-0.050332399782713755, 0.031944983002028124, 0.005757153247374316]
renderView1.CameraFocalPoint = [0.020720326341688633, 0.010470669716596603, 0.010644147405400872]
renderView1.CameraViewUp = [0.004949270345919701, 0.2374388307550575, 0.9713898838122179]
renderView1.CameraParallelScale = 0.0030018986837840496

#### uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).
