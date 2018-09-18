import vtk

def writeObjects(graph,
                 node_scalar_list = [],
                 edge_scalar_list = [],
                 node_label = '',
                 edge_label = '',
                 method = 'vtkPolyData',
                 fileout = 'test'):
    """
    Store points and/or graphs as vtkPolyData or vtkUnstructuredGrid.
    Required argument:
    - graph networkx graph
    Optional arguments:
    - node_scalar_list list of node scalars to include
    - edge_scalar_list is a list of edge salalrs to include
    - node_label is the string label of a string variable at each node
    - edge_label is the string label of a string variable at each edge
    - method = 'vtkPolyData' or 'vtkUnstructuredGrid'
    - fileout is the output file name (will be given .vtp or .vtu extension)
    """

    #node section
    points = vtk.vtkPoints()
    n_nodes = graph.number_of_nodes()
    points.SetNumberOfPoints(n_nodes)

    if node_scalar_list:
        node_arrays = {}
        for scalar_name in node_scalar_list:
            attribute = vtk.vtkDoubleArray()
            attribute.SetName(scalar_name)
            attribute.SetNumberOfComponents(1)
            attribute.SetNumberOfTuples(n_nodes)
            node_arrays[scalar_name] = attribute

    if node_label:
        label_nodes = vtk.vtkStringArray()
        label_nodes.SetName(node_label)
        label_nodes.SetNumberOfValues(n_nodes)

    #assign nodes
    for node in graph.nodes():
        node_data = graph.nodes(data=True)[node]
        #print(node)
        points.SetPoint(node, node_data['vertex'])
        if node_scalar_list:
            for scalar_name in node_scalar_list:
                node_arrays[scalar_name].SetValue(node, node_data[scalar_name])
        if node_label:
            label_nodes.SetValue(node, node_data[node_label])

    #edge section
    n_edges = graph.number_of_edges()
    if edge_scalar_list:
        edge_arrays = {}
        for scalar_name in edge_scalar_list:
            attribute = vtk.vtkDoubleArray()
            attribute.SetName(scalar_name)
            attribute.SetNumberOfComponents(1)
            attribute.SetNumberOfTuples(n_edges)
            edge_arrays[scalar_name] = attribute
    if edge_label:
        label_edges = vtk.vtkStringArray()
        label_edges.SetName(edge_label)
        label_edges.SetNumberOfValues(n_edges)

    # assign edge info
    if (n_edges > 0):
        lines = vtk.vtkCellArray()
        lines.Allocate(n_edges)
        for edge in graph.edges():
            edge_data = graph.edges[edge]
            cell_id = lines.InsertNextCell(2)
            lines.InsertCellPoint(edge[0])
            lines.InsertCellPoint(edge[1])   # line from point edge[0] to point edge[1]
            if edge_scalar_list:
                for scalar_name in edge_scalar_list:
                    edge_arrays[scalar_name].SetValue(cell_id, edge_data[scalar_name])
            if edge_label:
                label_edges.SetValue(cell_id, edge_data[edge_label])


    if method == 'vtkPolyData':
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        if (n_edges > 0):
            polydata.SetLines(lines)
        if node_scalar_list:
            for scalar_name in node_scalar_list:
                polydata.GetPointData().AddArray(node_arrays[scalar_name])
        if edge_scalar_list:
            for scalar_name in edge_scalar_list:
                polydata.GetCellData().AddArray(edge_arrays[scalar_name])
        if node_label:
            polydata.GetPointData().AddArray(label_nodes)
        if edge_label:
            polydata.GetCellData().AddArray(label_edges)
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(fileout+'.vtp')
        writer.SetInputData(polydata)
        writer.Write()
    elif method == 'vtkUnstructuredGrid':
        # caution: ParaView's Tube filter does not work on vtkUnstructuredGrid
        grid = vtk.vtkUnstructuredGrid()
        grid.SetPoints(points)
        if (n_edges > 0):
            grid.SetCells(vtk.VTK_LINE, line)
        if node_scalar_list:
            for scalar_name in node_scalar_list:
                grid.GetPointData().AddArray(node_arrays[scalar_name])
        if edge_scalar_list:
            for scalar_name in edge_scalar_list:
                grid.GetCellData().AddArray(edge_arrays[scalar_name])
        if node_label:
            grid.GetPointData().AddArray(label_nodes)
        if edge_label:
            grid.GetCellData().AddArray(label_edges)

        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(fileout+'.vtu')
        writer.SetInputData(grid)
        writer.Write()

def writePolyLine(graph, node_lists,
                 node_scalar_list = [],
                 edge_scalar_list = [],
                 node_label = '',
                 edge_label = '',
                 fileout = 'test'):
    """
    Store points and/or graphs as vtkPolyData or vtkUnstructuredGrid.
    Required argument:
    - graph networkx graph
    - ordered lists of nodes for each polyline
    Optional arguments:
    - node_scalar_list list of node scalars to include
    - edge_scalar_list is a list of edge salalrs to include
    - node_label is the string label of a string variable at each node
    - edge_label is the string label of a string variable at each edge
    - method = 'vtkPolyData' or 'vtkUnstructuredGrid'
    - fileout is the output file name (will be given .vtp or .vtu extension)
    """

    #node section
    points = vtk.vtkPoints()
    n_nodes = graph.number_of_nodes()
    points.SetNumberOfPoints(n_nodes)

    if node_scalar_list:
        node_arrays = {}
        for scalar_name in node_scalar_list:
            attribute = vtk.vtkDoubleArray()
            attribute.SetName(scalar_name)
            attribute.SetNumberOfComponents(1)
            attribute.SetNumberOfTuples(n_nodes)
            node_arrays[scalar_name] = attribute

    if node_label:
        label_nodes = vtk.vtkStringArray()
        label_nodes.SetName(node_label)
        label_nodes.SetNumberOfValues(n_nodes)

    #assign nodes
    for node in graph.nodes():
        node_data = graph.nodes(data=True)[node]
        #print(node)
        points.SetPoint(node, node_data['vertex'])
        if node_scalar_list:
            for scalar_name in node_scalar_list:
                node_arrays[scalar_name].SetValue(node, node_data[scalar_name])
        if node_label:
            label_nodes.SetValue(node, node_data[node_label])

    #edge section
    # n_edges = graph.number_of_edges()
    # if edge_scalar_list:
    #     edge_arrays = {}
    #     for scalar_name in edge_scalar_list:
    #         attribute = vtk.vtkDoubleArray()
    #         attribute.SetName(scalar_name)
    #         attribute.SetNumberOfComponents(1)
    #         attribute.SetNumberOfTuples(n_edges)
    #         edge_arrays[scalar_name] = attribute
    # if edge_label:
    #     label_edges = vtk.vtkStringArray()
    #     label_edges.SetName(edge_label)
    #     label_edges.SetNumberOfValues(n_edges)

    # assign edge info
    if node_lists:
        lines = vtk.vtkCellArray()
        lines.Allocate(len(node_lists))
        for line in node_lists:
            polyLine = vtk.vtkPolyLine()
            polyLine.GetPointIds().SetNumberOfIds(len(line))
            for idx, node in enumerate(line):
                polyLine.GetPointIds().SetId(idx,node)
            lines.InsertNextCell(polyLine)

        # for edge in graph.edges():
        #     edge_data = graph.edges[edge]
        #     cell_id = lines.InsertNextCell(2)
        #     lines.InsertCellPoint(edge[0])
        #     lines.InsertCellPoint(edge[1])   # line from point edge[0] to point edge[1]
        #     if edge_scalar_list:
        #         for scalar_name in edge_scalar_list:
        #             edge_arrays[scalar_name].SetValue(cell_id, edge_data[scalar_name])
        #     if edge_label:
        #         label_edges.SetValue(cell_id, edge_data[edge_label])


    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    if node_lists:
        polydata.SetLines(lines)
    if node_scalar_list:
        for scalar_name in node_scalar_list:
            polydata.GetPointData().AddArray(node_arrays[scalar_name])
    # if edge_scalar_list:
    #     for scalar_name in edge_scalar_list:
    #         polydata.GetCellData().AddArray(edge_arrays[scalar_name])
    if node_label:
        polydata.GetPointData().AddArray(label_nodes)
    # if edge_label:
    #     polydata.GetCellData().AddArray(label_edges)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(fileout+'.vtp')
    writer.SetInputData(polydata)
    writer.Write()
