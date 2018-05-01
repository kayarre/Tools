
#!/usr/bin/env python

import vtk
import numpy as np
from vmtk import vmtkscripts
import vmtk
import argparse
import copy

# extract the surface from unstructured mesh 
def run_script(args):
    print("extract dome from cfd")
    
    reader_dome = vmtkscripts.vmtkSurfaceReader()
    reader_dome.InputFileName = args.surface_file
    reader_dome.Execute()
    dome_surface = reader_dome.Surface
    
    mesh_reader = vmtkscripts.vmtkMeshReader()
    mesh_reader.InputFileName = args.mesh_file
    mesh_reader.Execute()
    
    mesh2surf = vmtkscripts.vmtkMeshToSurface()
    mesh2surf.Mesh = mesh_reader.Mesh
    mesh2surf.CleanOutput = 0
    mesh2surf.Execute()
    
    scale_cfd = vmtkscripts.vmtkSurfaceScaling()
    scale_cfd.ScaleFactor = 1000 # meters to mm
    scale_cfd.Surface = mesh2surf.Surface
    scale_cfd.Execute()
    
    
    dist = vmtkscripts.vmtkSurfaceDistance()
    dist.Surface = scale_cfd.Surface
    dist.ReferenceSurface = dome_surface
    dist.DistanceArrayName = "distance"
    dist.DistanceVectorsArrayName = "distance_vectors"
    dist.SignedDistanceArrayName = "signed_distance"
    dist.Execute()
    

    clip_copy = vtk.vtkPolyData()
    clip_copy.DeepCopy(dist.Surface)
    
    clip = vmtkscripts.vmtkSurfaceClipper()
    clip.Surface = dist.Surface
    clip.Interactive = 0
    clip.InsideOut = 1
    clip.ClipArrayName = "distance"
    clip.ClipValue = 0.1
    clip.Execute()
    
    conn = vmtkscripts.vmtkSurfaceConnectivity()
    conn.Surface = clip.Surface
    conn.Method ="largest"
    conn.CleanOutput = 0
    conn.Execute()
    
    normals = vmtkscripts.vmtkSurfaceNormals()
    normals.Surface = conn.Surface
    #accept defaults
    normals.Execute()

    writer = vmtkscripts.vmtkSurfaceWriter()
    writer.OutputFileName = args.file_out
    writer.Input = normals.Surface
    writer.Execute()
    
    if (args.inverse_out):
        clip_inverse = vmtkscripts.vmtkSurfaceClipper()
        clip_inverse.Surface = clip_copy
        clip_inverse.Interactive = 0
        clip_inverse.InsideOut = 0
        clip_inverse.ClipArrayName = "distance"
        clip_inverse.ClipValue = 0.1
        clip_inverse.Execute()
        
        conn2 = vmtkscripts.vmtkSurfaceConnectivity()
        conn2.Surface = clip_inverse.Surface
        conn2.Method ="all"
        conn2.CleanOutput = 0
        conn2.Execute()

        writer2 = vmtkscripts.vmtkSurfaceWriter()
        writer2.OutputFileName = args.inverse_out
        writer2.Input = conn2.Surface
        writer2.Execute()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='average probed information along lines')
    parser.add_argument("-i", dest="mesh_file", required=True, help="input mesh file", metavar="FILE")
    parser.add_argument("-d", dest="surface_file", required=True, help="input dome surface", metavar="FILE")

    parser.add_argument("-o", dest="file_out", required=True, help="output file clipped surface", metavar="FILE")
    parser.add_argument("--inverse", dest="inverse_out", required=False, help="output file of inverse clipped surface", metavar="FILE")
    args = parser.parse_args()
    #print(args)
    run_script(args)
