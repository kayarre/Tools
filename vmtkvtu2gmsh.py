# -*- coding: utf-8 -*-
#
'''
I/O for Gmsh's msh format, cf.
<http://gmsh.info//doc/texinfo/gmsh.html#File-formats>.

.. moduleauthor:: Nico Schlömer <nico.schloemer@gmail.com>
'''

import logging
import struct
import meshio
import numpy
import copy

def raw_from_cell_data(cell_data):
    # merge cell data
    cell_data_raw = {}
    for d in cell_data.values():
        for name, values in d.items():
            if name in cell_data_raw:
                cell_data_raw[name].append(values)
            else:
                cell_data_raw[name] = [values]
    for name in cell_data_raw:
        cell_data_raw[name] = numpy.concatenate(cell_data_raw[name])

    return cell_data_raw

num_nodes_per_cell = {
    'vertex': 1,
    'line': 2,
    'triangle': 3,
    'quad': 4,
    'quad8': 8,
    'tetra': 4,
    'hexahedron': 8,
    'hexahedron20': 20,
    'wedge': 6,
    'pyramid': 5,
    #
    'line3': 3,
    'triangle6': 6,
    'quad9': 9,
    'tetra10': 10,
    'hexahedron27': 27,
    'prism18': 18,
    'pyramid14': 14,
    #
    'line4': 4,
    'triangle10': 10,
    'quad16': 16,
    'tetra20': 20,
    'hexahedron64': 64,
    #
    'line5': 5,
    'triangle15': 15,
    'quad25': 25,
    'tetra35': 35,
    'hexahedron125': 125,
    #
    'line6': 6,
    'triangle21': 21,
    'quad36': 36,
    'tetra56': 56,
    'hexahedron216': 216,
    }

# Translate meshio types to gmsh codes
# http://gmsh.info//doc/texinfo/gmsh.html#MSH-ASCII-file-format
_gmsh_to_meshio_type = {
        1: 'line',
        2: 'triangle',
        3: 'quad',
        4: 'tetra',
        5: 'hexahedron',
        6: 'wedge',
        7: 'pyramid',
        8: 'line3',
        9: 'triangle6',
        10: 'quad9',
        11: 'tetra10',
        12: 'hexahedron27',
        13: 'prism18',
        14: 'pyramid14',
        15: 'vertex',
        16: 'quad8',
        17: 'hexahedron20',
        21: 'triangle10',
        23: 'triangle15',
        25: 'triangle21',
        26: 'line4',
        27: 'line5',
        28: 'line6',
        29: 'tetra20',
        30: 'tetra35',
        31: 'tetra56',
        36: 'quad16',
        37: 'quad25',
        38: 'quad36',
        92: 'hexahedron64',
        93: 'hexahedron125',
        94: 'hexahedron216',
        }
_meshio_to_gmsh_type = {v: k for k, v in _gmsh_to_meshio_type.items()}


def _write_physical_names(fh, field_data):
    # Write physical names
    entries = []
    for phys_name in field_data:
        try:
            phys_num, phys_dim = field_data[phys_name]
            phys_num, phys_dim = int(phys_num), int(phys_dim)
            entries.append((phys_dim, phys_num, phys_name))
        except (ValueError, TypeError):
            logging.warning(
                'Field data contains entry that cannot be processed.'
            )
    entries.sort()
    if entries:
        fh.write('$PhysicalNames\n'.encode('utf-8'))
        fh.write('{}\n'.format(len(entries)).encode('utf-8'))
        for entry in entries:
            fh.write('{} {} "{}"\n'.format(*entry).encode('utf-8'))
        fh.write('$EndPhysicalNames\n'.encode('utf-8'))
    return


def _write_nodes(fh, points, write_binary):
    fh.write('$Nodes\n'.encode('utf-8'))
    fh.write('{}\n'.format(len(points)).encode('utf-8'))
    if write_binary:
        dtype = [('index', numpy.int32), ('x', numpy.float64, (3,))]
        tmp = numpy.empty(len(points), dtype=dtype)
        tmp['index'] = 1 + numpy.arange(len(points))
        tmp['x'] = points
        fh.write(tmp.tostring())
        fh.write('\n'.encode('utf-8'))
    else:
        for k, x in enumerate(points):
            fh.write(
                '{} {!r} {!r} {!r}\n'.format(k+1, x[0], x[1], x[2])
                .encode('utf-8')
                )
    fh.write('$EndNodes\n'.encode('utf-8'))
    return


def _write_elements(fh, cells, write_binary, cell_tags=None):
    # write elements
    fh.write('$Elements\n'.encode('utf-8'))
    # count all cells
    total_num_cells = sum([data.shape[0] for _, data in cells.items()])
    fh.write('{}\n'.format(total_num_cells).encode('utf-8'))

    cell_tags = {} if cell_tags is None else cell_tags

    consecutive_index = 0
    for cell_type, node_idcs in cells.items():
        if cell_type in cell_tags and cell_tags[cell_type]:
            for key in cell_tags[cell_type]:
                # assert data consistency
                assert len(cell_tags[cell_type][key]) == len(node_idcs)
                # TODO assert that the data type is int

            # if a tag is present, make sure that there are 'physical' and
            # 'geometrical' as well.
            if 'physical' not in cell_tags[cell_type]:
                cell_tags[cell_type]['physical'] = \
                    numpy.ones(len(node_idcs), dtype=numpy.int32)
            if 'geometrical' not in cell_tags[cell_type]:
                cell_tags[cell_type]['geometrical'] = \
                    numpy.ones(len(node_idcs), dtype=numpy.int32)

            # 'physical' and 'geometrical' go first; this is what the gmsh
            # file format prescribes
            keywords = list(cell_tags[cell_type].keys())
            keywords.remove('physical')
            keywords.remove('geometrical')
            sorted_keywords = ['physical', 'geometrical'] + keywords
            fcd = numpy.column_stack([
                    cell_tags[cell_type][key] for key in sorted_keywords
                    ])
        else:
            #no cell data
            fcd = numpy.empty([len(node_idcs), 0], dtype=numpy.int32)

        if write_binary:
            # header
            fh.write(struct.pack('i', _meshio_to_gmsh_type[cell_type]))
            fh.write(struct.pack('i', node_idcs.shape[0]))
            fh.write(struct.pack('i', fcd.shape[1]))
            # actual data
            a = numpy.arange(
                len(node_idcs), dtype=numpy.int32
                )[:, numpy.newaxis]
            a += 1 + consecutive_index
            array = numpy.hstack([a, fcd, node_idcs + 1])
            fh.write(array.tostring())
        else:
            form = '{} ' + str(_meshio_to_gmsh_type[cell_type]) \
                + ' ' + str(fcd.shape[1]) \
                + ' {} {}\n'
            for k, c in enumerate(node_idcs):
                fh.write(
                    form.format(
                        consecutive_index + k + 1,
                        ' '.join([str(val) for val in fcd[k]]),
                        ' '.join([str(cc + 1) for cc in c])
                        ).encode('utf-8')
                    )

        consecutive_index += len(node_idcs)
    if write_binary:
        fh.write('\n'.encode('utf-8'))
    fh.write('$EndElements\n'.encode('utf-8'))
    return


def _write_data(fh, tag, name, data, write_binary):
    fh.write('${}\n'.format(tag).encode('utf-8'))
    # <http://gmsh.info/doc/texinfo/gmsh.html>:
    # > Number of string tags.
    # > gives the number of string tags that follow. By default the first
    # > string-tag is interpreted as the name of the post-processing view and
    # > the second as the name of the interpolation scheme. The interpolation
    # > scheme is provided in the $InterpolationScheme section (see below).
    fh.write('{}\n'.format(1).encode('utf-8'))
    fh.write('{}\n'.format(name).encode('utf-8'))
    fh.write('{}\n'.format(1).encode('utf-8'))
    fh.write('{}\n'.format(0.0).encode('utf-8'))
    # three integer tags:
    fh.write('{}\n'.format(3).encode('utf-8'))
    # time step
    fh.write('{}\n'.format(0).encode('utf-8'))
    # number of components
    num_components = data.shape[1] if len(data.shape) > 1 else 1
    assert num_components in [1, 3, 9], \
        'Gmsh only permits 1, 3, or 9 components per data field.'
    fh.write('{}\n'.format(num_components).encode('utf-8'))
    # num data items
    fh.write('{}\n'.format(data.shape[0]).encode('utf-8'))
    # actually write the data
    if write_binary:
        dtype = [
            ('index', numpy.int32),
            ('data', numpy.float64, num_components)
            ]
        tmp = numpy.empty(len(data), dtype=dtype)
        tmp['index'] = 1 + numpy.arange(len(data))
        tmp['data'] = data
        fh.write(tmp.tostring())
        fh.write('\n'.encode('utf-8'))
    else:
        fmt = ' '.join(['{}'] + ['{!r}'] * num_components) + '\n'
        # TODO unify
        if num_components == 1:
            for k, x in enumerate(data):
                fh.write(fmt.format(k+1, x).encode('utf-8'))
        else:
            for k, x in enumerate(data):
                fh.write(fmt.format(k+1, *x).encode('utf-8'))

    fh.write('$End{}\n'.format(tag).encode('utf-8'))
    return


def write(
        filename,
        points,
        cells,
        point_data=None,
        cell_data=None,
        field_data=None,
        cell_tags=None,
        write_binary=True,
        ):
    '''Writes msh files, cf.
    <http://gmsh.info//doc/texinfo/gmsh.html#MSH-ASCII-file-format>.
    '''
    point_data = {} if point_data is None else point_data
    cell_data = {} if cell_data is None else cell_data
    field_data = {} if field_data is None else field_data
    cell_tags = {} if cell_tags is None else cell_tags

    if write_binary:
        for key in cells:
            if cells[key].dtype != numpy.int32:
                logging.warning(
                    'Binary Gmsh needs 32-bit integers (got %s). Converting.',
                    cells[key].dtype
                    )
                cells[key] = numpy.array(cells[key], dtype=numpy.int32)

    with open(filename, 'wb') as fh:
        mode_idx = 1 if write_binary else 0
        size_of_double = 8
        fh.write((
            '$MeshFormat\n2.2 {} {}\n'.format(mode_idx, size_of_double)
            ).encode('utf-8'))
        if write_binary:
            fh.write(struct.pack('i', 1))
            fh.write('\n'.encode('utf-8'))
        fh.write('$EndMeshFormat\n'.encode('utf-8'))

        if field_data:
            _write_physical_names(fh, field_data)

        _write_nodes(fh, points, write_binary)
        _write_elements(fh, cells, write_binary, cell_tags)
        for name, dat in point_data.items():
            _write_data(fh, 'NodeData', name, dat, write_binary)
        cell_data_raw = raw_from_cell_data(cell_data)
        for name, dat in cell_data_raw.items():
            _write_data(fh, 'ElementData', name, dat, write_binary)

    return






if ( __name__ == '__main__' ):
    fn = "/home/ksansom/caseFiles/ultrasound/cases/DSI020CALb/vmtk/DSI020CALb_vmtk_decimate_trim_ext2_mesh.vtu"

    fn_out = "/home/ksansom/caseFiles/ultrasound/cases/DSI020CALb/vmtk/DSI020CALb_gmsh_flipwedge.msh"
    #fn_out = "/home/ksansom/caseFiles/ultrasound/cases/DSI020CALb/vmtk/DSI020CALb_vmtk_decimate_trim_ext2_mesh.msh"

    points, cells, point_data, cell_data, field_data = meshio.read(fn, file_format="vtu-binary")
    el_2_bnd = {}
    bnd_ids = []
    for eltype in cells.keys():
        print(eltype, numpy.unique(cell_data[eltype]['CellEntityIds']))
        el_2_bnd[eltype] = list(numpy.unique(cell_data[eltype]['CellEntityIds']))
        bnd_ids = list(set().union(bnd_ids, el_2_bnd[eltype]))
    bnd_ids = sorted(bnd_ids)
    print(bnd_ids)

    inv_map = {}
    for k, v in el_2_bnd.items():
        for i in v:
            if (i in inv_map.keys()):
                inv_map[i].append(k)
            else:
                inv_map[i] = [k]

    print(inv_map)
    cell_tags = {}
    for eltype in cells.keys():
        geo = copy.deepcopy(cell_data[eltype]['CellEntityIds'])
        phy = copy.deepcopy(cell_data[eltype]['CellEntityIds'])
        if (eltype not in field_data.keys()):
            cell_tags[eltype] = {}
        cell_tags[eltype]['geometrical'] =  geo
        cell_tags[eltype]['physical'] = phy

    for eltype in cells.keys():
        print(eltype, numpy.unique(cell_tags[eltype]['geometrical']))
    cnt2 = 0

    physical_names = {}
    for bnd_id in bnd_ids:
        el_types = inv_map[bnd_id] # get the element types for that boundary
        for eltype in el_types:
            idx_  = numpy.where(cell_data[eltype]['CellEntityIds'] == bnd_id)
            if (eltype == "tetra" and bnd_id == 0):
                cell_tags[eltype]['geometrical'][idx_] = 1
                cell_tags[eltype]['physical'][idx_] = 1
                physical_names["tetra"] = [1]
            elif(eltype == "wedge" and bnd_id == 0):
                cell_tags[eltype]['geometrical'][idx_] = 0
                cell_tags[eltype]['physical'][idx_] = 1
                physical_names["wedge"] = [1]
                cnt2 += 1
            elif(eltype == "triangle" and bnd_id == 1):
                cell_tags[eltype]['geometrical'][idx_] = 2
                cell_tags[eltype]['physical'][idx_] = 2
                physical_names["wall"] = [3]
            else:
                cur_idx = bnd_id + cnt2
                cell_tags[eltype]['geometrical'][idx_] = cur_idx
                cell_tags[eltype]['physical'][idx_] = cur_idx
                if ( "flow" not in physical_names.keys()):
                    physical_names["flow"] = []
                if (cur_idx+1 not in physical_names["flow"]):
                    physical_names["flow"].append(cur_idx+1)
            #cell_tags[eltype]['physical'] =  1+copy.deepcopy(cell_tags[eltype]['geometrical'])

    print(physical_names)
    #print(new_cell_data)
    for eltype in cells.keys():
        print(eltype, numpy.unique(cell_tags[eltype]['geometrical']))

    # node ordering appears to be wrong
    #cells['wedge'] = numpy.fliplr(cells['wedge'])
    #cells['wedge'] = numpy.roll(cells['wedge'], 3, axis =1)
    cells['wedge'] = numpy.fliplr(numpy.roll(cells['wedge'], 3, axis =1))

    # write it out
    write(fn_out, points, cells,
         point_data=point_data,
         cell_data=cell_data,
         field_data=field_data,
         cell_tags=cell_tags,
         write_binary=False
         )
