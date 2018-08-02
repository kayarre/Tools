#!/usr/bin/env python

import numpy as np
from vmtk import vmtkscripts
import argparse




def Execute(args):
    print("flip boundary region for flow extensions")
    
    flip_id = [int(i) for i in args.flip_ids.strip(" ").split(",")]   
    print(flip_id)
    surface_reader = vmtkscripts.vmtkSurfaceReader()
    surface_reader.InputFileName = args.surface_file
    surface_reader.Execute()
    
    surf = surface_reader.Surface

    for i in flip_id:
        id_ = surf.GetCellData().GetArray("BoundaryRegion").GetTuple(i)
        print(id_)
        new = [ (j+1)%2 for j in id_]
        print(new)
        surf.GetCellData().GetArray("BoundaryRegion").SetTuple(i, new)
        
    
    writer = vmtkscripts.vmtkSurfaceWriter()
    writer.OutputFileName = args.out_file
    writer.Input = surf
    writer.Execute()




if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Clip flow extensions')
    parser.add_argument("-i", dest="surface_file", required=True, help="vertex data of boundary information", metavar="FILE")
    parser.add_argument("-o", dest="out_file", required=True,
                        help="output filename for vertex data", metavar="FILE")
    parser.add_argument("--ids", dest="flip_ids", type=str, help="string of comma separated values with cell ids to invert", metavar="str")
    args = parser.parse_args()
    #print(args)
    Execute(args)
