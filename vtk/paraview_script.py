#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'XML Unstructured Grid Reader'
calc_test_node_statsvtu = XMLUnstructuredGridReader(FileName=['/raid/home/ksansom/caseFiles/mri/VWI_proj/case4/fluent_dsa/vtk_out/calc_test_node_stats.vtu'])
calc_test_node_statsvtu.PointArrayStatus = ['WSS', 'WSSG', 'absolute_pressure', 'TAWSS', 'TAWSSG', 'OSI', 'velocity']

# create a new 'XML Unstructured Grid Reader'
calc_test_node2vtu = XMLUnstructuredGridReader(FileName=['/raid/home/ksansom/caseFiles/mri/VWI_proj/case4/fluent_dsa/vtk_out/calc_test_node.vtu'])
calc_test_node2vtu.PointArrayStatus = ['absolute_pressure', 'velocity', 'x_velocity', 'x_wall_shear', 'y_velocity', 'y_wall_shear', 'z_velocity', 'z_wall_shear', 'WSS', 'x_WSS_grad', 'y_WSS_grad', 'z_WSS_grad', 'WSSG']

# get animation scene
animationScene1 = GetAnimationScene()

# update animation scene based on data timesteps
animationScene1.UpdateAnimationUsingDataTimeSteps()

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
# renderView1.ViewSize = [1431, 835]

# reset view to fit data
renderView1.ResetCamera()

# get color transfer function/color map for 'OSI'
oSILUT = GetColorTransferFunction('OSI')
oSILUT.RGBPoints = [-18.349878464749807, 0.231373, 0.298039, 0.752941, -8.963760541398452, 0.865003, 0.865003, 0.865003, 0.42235738195290295, 0.705882, 0.0156863, 0.14902]
oSILUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'OSI'
oSIPWF = GetOpacityTransferFunction('OSI')
oSIPWF.Points = [-18.349878464749807, 0.0, 0.5, 0.0, 0.42235738195290295, 1.0, 0.5, 0.0]
oSIPWF.ScalarRangeInitialized = 1

# show data in view
calc_test_node_statsvtuDisplay = Show(calc_test_node_statsvtu, renderView1)

# reset view to fit data
renderView1.ResetCamera()

# show color bar/color legend
calc_test_node_statsvtuDisplay.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'WSSG'
wSSGLUT = GetColorTransferFunction('WSSG')
wSSGLUT.RGBPoints = [0.4383581847104034, 0.231373, 0.298039, 0.752941, 12857.450125922955, 0.865003, 0.865003, 0.865003, 25714.4618936612, 0.705882, 0.0156863, 0.14902]
wSSGLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'WSSG'
wSSGPWF = GetOpacityTransferFunction('WSSG')
wSSGPWF.Points = [0.4383581847104034, 0.0, 0.5, 0.0, 25714.4618936612, 1.0, 0.5, 0.0]
wSSGPWF.ScalarRangeInitialized = 1

# show data in view
calc_test_node2vtuDisplay = Show(calc_test_node2vtu, renderView1)

# show color bar/color legend
calc_test_node2vtuDisplay.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'z_wall_shear'
z_wall_shearLUT = GetColorTransferFunction('z_wall_shear')
z_wall_shearLUT.RGBPoints = [-3.4691720008850098, 0.231373, 0.298039, 0.752941, -0.5872918367385864, 0.865003, 0.865003, 0.865003, 2.294588327407837, 0.705882, 0.0156863, 0.14902]
z_wall_shearLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'z_wall_shear'
z_wall_shearPWF = GetOpacityTransferFunction('z_wall_shear')
z_wall_shearPWF.Points = [-3.4691720008850098, 0.0, 0.5, 0.0, 2.294588327407837, 1.0, 0.5, 0.0]
z_wall_shearPWF.ScalarRangeInitialized = 1


# update the view to ensure updated data information
renderView1.Update()


# show data in view
calc_test_node2vtuDisplay = Show(calc_test_node2vtu, renderView1)

# show color bar/color legend
calc_test_node2vtuDisplay.SetScalarBarVisibility(renderView1, False)

# hide data in view
Hide(calc_test_node_statsvtu, renderView1)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
wSSGLUT.ApplyPreset('Preset', True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
wSSGPWF.ApplyPreset('Preset', True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
wSSGLUT.ApplyPreset('Preset', True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
wSSGPWF.ApplyPreset('Preset', True)

# set active source
SetActiveSource(calc_test_node_statsvtu)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
oSILUT.ApplyPreset('Preset', True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
oSIPWF.ApplyPreset('Preset', True)

# create a new 'Extract Surface'
extractSurface1 = ExtractSurface(Input=calc_test_node_statsvtu)

# show data in view
extractSurface1Display = Show(extractSurface1, renderView1)
extractSurface1Display.Opacity = 0.3

# hide data in view
Hide(calc_test_node_statsvtu, renderView1)

# show color bar/color legend
extractSurface1Display.SetScalarBarVisibility(renderView1, False)

# update the view to ensure updated data information
renderView1.Update()

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(oSILUT, renderView1)

# hide data in view
Hide(extractSurface1, renderView1)

# hide data in view
Hide(calc_test_node_statsvtu, renderView1)
Hide(calc_test_node2vtu, renderView1)

# create a new 'Clip'
clip1 = Clip(Input=calc_test_node_statsvtu)
clip1.ClipType = 'Plane'
clip1.Scalars = ['POINTS', 'OSI']
clip1.Value = -8.963760541398452
clip1.ClipType.Origin = [0.019185381077859477, 0.008678421124464417, 0.019721650218845817]
clip1.ClipType.Normal = [-0.9140283519495734, -0.03315688808077104, 0.40429295393952386]
clip1.InsideOut = 0

# show data in view
clip1Display = Show(clip1, renderView1)

# show color bar/color legend
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

# show data in view
clip2Display = Show(clip2, renderView1)
# show color bar/color legend
clip2Display.SetScalarBarVisibility(renderView1, False)
# hide data in view
Hide(clip2, renderView1)

# create a new 'Clip'
clip3 = Clip(Input=clip2)
clip3.ClipType = 'Plane'
clip3.Scalars = ['POINTS', 'OSI']
clip3.Value = -4.334532492423367

# init the 'Plane' selected for 'ClipType'
clip3.ClipType.Origin = [0.016755502272172117, 0.009108265077504313, 0.02113123859407696]
clip3.ClipType.Normal = [-0.08027021454652083, 0.9370193430035909, -0.33992858587323105]

# show data in view
clip3Display = Show(clip3, renderView1)

# show color bar/color legend
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

# show data in view
clip4Display = Show(clip4, renderView1)

# show color bar/color legend
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
clip5.ClipType.Axis = [0.32150296771440057, 0.7986387618253851, -0.5087356581377493]
clip5.ClipType.Radius = 0.0021806721886416867
# toggle 3D widget visibility (only when running from the GUI)
Show3DWidgets(proxy=clip5.ClipType)

# show data in view
clip5Display = Show(clip5, renderView1)


# show color bar/color legend
clip5Display.SetScalarBarVisibility(renderView1, False)

# update the view to ensure updated data information
renderView1.Update()

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
# Properties modified on slice1.SliceType
slice1.SliceType.Origin = [0.01406339196240306, 0.011922889508895057, 0.02007914092262216]
slice1.SliceType.Normal = [0.9166910278672833, -0.3166687289041705, 0.2437180246962479]
slice1.SliceType.Offset = 0.0

# show data in view
slice1Display = Show(slice1, renderView1)


# show color bar/color legend
slice1Display.SetScalarBarVisibility(renderView1, False)

# update the view to ensure updated data information
renderView1.Update()

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

# show color bar/color legend
tube1Display.SetScalarBarVisibility(renderView1, False)

renderView1.Update()

# set active source
SetActiveSource(slice1)

# get color transfer function/color map for 'OSI'
oSILUT = GetColorTransferFunction('OSI')

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
# renderView1.ViewSize = [1431, 835]

# reset view to fit data
renderView1.ResetCamera()

# update the view to ensure updated data information
renderView1.Update()


# create a new 'Clip'
clip1_2 = Clip(Input=calc_test_node2vtu)
clip1_2.ClipType = 'Plane'
clip1_2.Scalars = ['POINTS', 'OSI']
clip1_2.Value = -8.963760541398452
clip1_2.ClipType.Origin = [0.019185381077859477, 0.008678421124464417, 0.019721650218845817]
clip1_2.ClipType.Normal = [-0.9140283519495734, -0.03315688808077104, 0.40429295393952386]
clip1_2.InsideOut = 0

# show data in view
clip1_2Display = Show(clip1_2, renderView1)

# show color bar/color legend
clip1_2Display.SetScalarBarVisibility(renderView1, False)

# hide data in view
Hide(clip1_2, renderView1)

# create a new 'Clip'
clip2_2 = Clip(Input=clip1_2)
clip2_2.ClipType = 'Plane'
clip2_2.Scalars = ['POINTS', 'OSI']
clip2_2.Value = -4.334532492423367
# init the 'Plane' selected for 'ClipType'
clip2_2.ClipType.Origin = [0.016463510161933183, 0.013672129568797606, 0.020951286955119792]
clip2_2.ClipType.Normal = [-0.14808092203454595, -0.9422076182376612, -0.30052761048581295]

# show data in view
clip2_2Display = Show(clip2_2, renderView1)
# show color bar/color legend
clip2_2Display.SetScalarBarVisibility(renderView1, False)
# hide data in view
Hide(clip2_2, renderView1)

# create a new 'Clip'
clip3_2 = Clip(Input=clip2_2)
clip3_2.ClipType = 'Plane'
clip3_2.Scalars = ['POINTS', 'OSI']
clip3_2.Value = -4.334532492423367

# init the 'Plane' selected for 'ClipType'
clip3_2.ClipType.Origin = [0.016755502272172117, 0.009108265077504313, 0.02113123859407696]
clip3_2.ClipType.Normal = [-0.08027021454652083, 0.9370193430035909, -0.33992858587323105]

# show data in view
clip3_2Display = Show(clip3_2, renderView1)

# show color bar/color legend
clip3_2Display.SetScalarBarVisibility(renderView1, False)
# hide data in view
Hide(clip3_2, renderView1)

# create a new 'Clip'
clip4_2 = Clip(Input=clip3_2)
clip4_2.ClipType = 'Plane'
clip4_2.Scalars = ['POINTS', 'OSI']
clip4_2.Value = -0.22316894760113207

# init the 'Plane' selected for 'ClipType'
clip4_2.ClipType.Origin = [0.013133594821495129, 0.009304044349573848, 0.020326088763650343]
clip4_2.ClipType.Normal = [0.5663667872989001, 0.8238781757097235, 0.02129351624181657]

# show data in view
clip4_2Display = Show(clip4_2, renderView1)

# show color bar/color legend
clip4_2Display.SetScalarBarVisibility(renderView1, False)
# hide data in view
Hide(clip4_2, renderView1)

# create a new 'Clip'
clip5_2 = Clip(Input=clip4_2)
clip5_2.ClipType = 'Cylinder'
clip5_2.Scalars = ['POINTS', 'OSI']
clip5_2.Value = -0.22316894760113207

# init the 'Plane' selected for 'ClipType'
clip5_2.ClipType.Center = [0.018777387893322797, 0.01057777265095785, 0.021172440223998248]
clip5_2.ClipType.Axis = [0.32150296771440057, 0.7986387618253851, -0.5087356581377493]
clip5_2.ClipType.Radius = 0.0021806721886416867
# toggle 3D widget visibility (only when running from the GUI)
Show3DWidgets(proxy=clip5_2.ClipType)

# show data in view
clip5_2Display = Show(clip5_2, renderView1)


# show color bar/color legend
clip5_2Display.SetScalarBarVisibility(renderView1, False)

# update the view to ensure updated data information
renderView1.Update()

# Rescale transfer function
oSILUT.RescaleTransferFunction(0.0, 0.5)

# Rescale transfer function
oSIPWF.RescaleTransferFunction(0.0, 0.5)

# hide data in view
Hide(clip5_2, renderView1)

# create a new 'Slice'
slice1_2 = Slice(Input=clip5_2)
slice1_2.SliceType = 'Plane'
slice1_2.SliceOffsetValues = [0.0]
# Properties modified on slice1.SliceType
slice1_2.SliceType.Origin = [0.01406339196240306, 0.011922889508895057, 0.02007914092262216]
slice1_2.SliceType.Normal = [0.9166910278672833, -0.3166687289041705, 0.2437180246962479]
slice1_2.SliceType.Offset = 0.0

# show data in view
slice1_2Display = Show(slice1_2, renderView1)


# show color bar/color legend
slice1_2Display.SetScalarBarVisibility(renderView1, False)

# update the view to ensure updated data information
renderView1.Update()

# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=slice1_2.SliceType)

# hide data in view
Hide(slice1_2, renderView1)

# create a new 'Tube'
tube1_2 = Tube(Input=slice1_2)
tube1_2.Scalars = ['POINTS', 'OSI']
tube1_2.Vectors = ['POINTS', 'velocity']
tube1_2.Radius = 5.0e-5
tube1_2.Capping = 0
tube1_2.NumberofSides = 8

# show data in view
tube1_2Display = Show(tube1_2, renderView1)

# show color bar/color legend
tube1_2Display.SetScalarBarVisibility(renderView1, False)

renderView1.Update()

# set active source
SetActiveSource(slice1_2)

# get color transfer function/color map for 'OSI'
oSILUT = GetColorTransferFunction('OSI')

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
# renderView1.ViewSize = [1431, 835]

# reset view to fit data
renderView1.ResetCamera()

# update the view to ensure updated data information
renderView1.Update()
