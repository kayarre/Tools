import vtk

reader = vtk.vtkXMLPolyDataReader()
reader.SetFileName("")

#Combine the two data sets
appendFilter = vrk.vtkAppendFilter()
appendFilter.AddInputConnection(reader.GetOutputPort())
appendFilter.Update()
unstructuredGrid = vtk.vtkUnstructuredGrid()
unstructuredGrid.ShallowCopy(appendFilter.GetOutput())

#Write the unstructured grid
writer = vtk.vtkXMLUnstructuredGridWriter()
writer.SetFileName("UnstructuredGrid.vtu")
writer.SetInputData(unstructuredGrid)
writer.Write()
