#!/usr/bin/env python
"""File format conversion
category: vtk, file conversion, tomb"""
import os, sys
import vtk

def vtk2vtp(invtkfile, outvtpfile, binary=False):
    """What it says on the label"""
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(invtkfile)
    reader.Update()

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(outvtpfile)
    if binary:
        writer.SetFileTypeToBinary()
    writer.SetInputData(reader.GetOutput())
    writer.Update()

def vtk2vtu_unstructured(invtkfile, outvtpfile, binary=False):
    """What it says on the label"""
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(invtkfile)
    reader.Update()

    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(outvtpfile)
    if binary:
        writer.SetFileTypeToBinary()
    else:
        writer.SetDataModeToAscii()
    writer.SetInputData(reader.GetOutput())
    writer.Update()


def vtk2vtp_unstructured(invtkfile, outvtpfile, binary=False):
    """exports surface"""
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(invtkfile)
    reader.Update()
    
    geometryFilter = vtk.vtkGeometryFilter()
    geometryFilter.SetInputData(reader.GetOutput())
    geometryFilter.Update()
    
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(outvtpfile)
    if binary:
        writer.SetFileTypeToBinary()
    else:
        writer.SetDataModeToAscii()
    writer.SetInputData(geometryFilter.GetOutput())
    writer.Update()


if __name__ == '__main__':
    args = sys.argv
    binary = False
    if '-b' in args:
        args.remove('-b')
        binary = True
    if len(args) < 2:
        print('Batch converts vtk files to vtp files.\nUsage:\n    vtk2vtp.py model1.vtk model2.vtk ...')
        print('    [-b] causes output to be in binary format, much smaller vtp file size, if it happens to work')
        sys.exit()
    infiles = args[1:]
    for vtkfile in infiles:
        if vtkfile[-4:] != '.vtk':
            print(vtkfile, "doesn't look like a vtk file, won't convert")
            continue
        #vtk2vtu_unstructured(vtkfile, vtkfile[:-4]+'.vtu', binary=binary)
        vtk2vtp_unstructured(vtkfile, vtkfile[:-4]+'.vtp', binary=binary)
        #vtk2vtp(vtkfile, vtkfile[:-4]+'.vtp', binary=binary)
