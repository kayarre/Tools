#! /usr/bin/env python3
'''
Convert a mesh file to another.
'''
from __future__ import print_function

import numpy
import vtk
import meshio


def _main():
    # Parse command line arguments.
    args = _parse_options()

    # read mesh data
    points, cells, point_data, cell_data, field_data = \
        meshio.read(args.infile, file_format="vtu-binary")

    print('Number of points: {}'.format(len(points)))
    print('Elements:')
    for tpe, elems in cells.items():
        print('  Number of {}s: {}'.format(tpe, len(elems)))

    if point_data:
        print('Point data: {}'.format(', '.join(point_data.keys())))

    cell_data_keys = set()
    for cell_type in cell_data:
        cell_data_keys = cell_data_keys.union(cell_data[cell_type].keys())
    if cell_data_keys:
        print('Cell data: {}'.format(', '.join(cell_data_keys)))

    if args.prune:
        cells.pop('vertex', None)
        cells.pop('line', None)
        if 'tetra' in cells:
            # remove_lower_order_cells
            cells.pop('triangle', None)
        # remove_orphaned_nodes.
        # find which nodes are not mentioned in the cells and remove them
        flat_cells = cells['tetra'].flatten()
        orphaned_nodes = numpy.setdiff1d(numpy.arange(len(points)), flat_cells)
        points = numpy.delete(points, orphaned_nodes, axis=0)
        # also adapt the point data
        for key in point_data:
            point_data[key] = numpy.delete(
                    point_data[key],
                    orphaned_nodes,
                    axis=0
                    )

        # reset GLOBAL_ID
        if 'GLOBAL_ID' in point_data:
            point_data['GLOBAL_ID'] = numpy.arange(1, len(points)+1)

        # We now need to adapt the cells too.
        diff = numpy.zeros(len(flat_cells), dtype=flat_cells.dtype)
        for orphan in orphaned_nodes:
            diff[numpy.argwhere(flat_cells > orphan)] += 1
        flat_cells -= diff
        cells['tetra'] = flat_cells.reshape(cells['tetra'].shape)

    # Some converters (like VTK) require `points` to be contiguous.
    points = numpy.ascontiguousarray(points)

    # write it out
    meshio.write(
        args.outfile,
        points,
        cells,
        file_format="gmsh-ascii",
        point_data=point_data,
        cell_data=cell_data,
        field_data=field_data
        )

    return


def _parse_options():
    '''Parse input options.'''
    import argparse

    parser = argparse.ArgumentParser(
            description=(
                'Convert vtu to gmsh mesh formats.'
                )
            )

    parser.add_argument(
        'infile',
        type=str,
        help='mesh file to be read from'
        )

    parser.add_argument(
        'outfile',
        type=str,
        help='mesh file to be written to'
        )

    parser.add_argument(
            '--prune', '-p',
            action='store_true',
            help='remove lower order cells, remove orphaned nodes'
            )

    parser.add_argument(
        '--version', '-v',
        action='version',
        version='%(prog)s ' + ('(version %s)' % meshio.__version__)
        )

    return parser.parse_args()


if __name__ == '__main__':
    # python read_vtu.py --input-format vtu-binary --output-format gmsh-binary /home/ksansom/caseFiles/ultrasound/cases/DSI020CALb/vmtk/DSI020CALb_vmtk_decimate_trim_ext2_mesh.vtu /home/ksansom/caseFiles/ultrasound/cases/DSI020CALb/vmtk/DSI020CALb_vmtk_decimate_trim_ext2_mesh.msh

    _main()










reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName(fn)
reader.Update()

#want to keep track of cellIds
ids_filter = vtk.vtkIdFilter()
ids_filter.SetInputConnection(reader.GetOutputPort())
#ids_filter.PointIdsOn()
ids_filter.CellIdsOn()
ids_filter.FieldDataOn()
ids_filter.SetIdsArrayName("Ids")
ids_filter.Update()

vtkMesh = ids_filter.GetOutput()

numberOfCellArrays = vtkMesh.GetCellData().GetNumberOfArrays()
cell_entity_id = 0
cell_id_id = 0
arrayNames = []
for i in range(numberOfCellArrays):
    arrayNames.append(vtkMesh.GetCellData().GetArrayName(i))
    if (arrayNames[-1] == "CellEntityIds"):
        cell_entity_id = i
    if (arrayNames[-1]  == "Ids"):
        cell_id_id = i

entity_range = [0., 0.]
vtkMesh.GetCellData().GetArray(cell_entity_id).GetRange(entity_range)

#//vtkCellArray *cells;
#//cells = vtkMesh->GetCells();

begin = int(entity_range[0])
end = int(entity_range[1])

thresh = vtk.vtkThreshold()
thresh.SetInputData(mesh)
thresh.SetInputArrayToProcess(1, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "CellEntityIds")

for j in range(begin, end+1):
    thresh.ThresholdBetween(j, j)

    thresh.Update()
    #pointIdArray = thresh.GetOutput().GetPointData().GetArray(ids_name)
    cellIdArray = thresh.GetOutput().GetCellData().GetArray(cell_id_id))
    for i in range(cellIdArray.GetNumberOfTuples()):
        #id of the cell
        cell_id_filt = static_cast<vtkIdType>(std::round(cellIdArray->GetComponent(q, 0)));
#     vtkMesh->GetCellPoints(cell_id_filt, npts, ptIds);
#     cell_type = vtkMesh->GetCellType(cell_id_filt);
#     vtkCellNumPoints = MapVtkCellType(cell_type, nekpp_type);
#
#     if (vtkCellNumPoints == -1)
#     {
#         std::cout << "nonsensical, empty cell" << std::endl;
#         continue;
#     }
#
#     for (j = 0; j < npts - vtkCellNumPoints + 1; ++j)
#     {
#         // Create element tags
#         vector<int> tags;
#         tags.push_back(int(q));     // composite
#         tags.push_back(nekpp_type); // element type
#
#         // Read element node list
#         vector<NodeSharedPtr> nodeList;
#         for (k = j; k < j + vtkCellNumPoints; ++k)
#         {
#             nodeList.push_back(m_mesh->m_node[ptIds[k]]);
#         }
#
#         // Create element
#         ElmtConfig conf(nekpp_type, 1, false, false);
#         ElementSharedPtr E = GetElementFactory().CreateInstance(
#             nekpp_type, conf, nodeList, tags);
#
#         // Determine mesh expansion dimension
#         if (E->GetDim() > m_mesh->m_expDim)
#         {
#             m_mesh->m_expDim = E->GetDim();
#         }
#         m_mesh->m_element[E->GetDim()].push_back(E);
#     }
# }
# }
