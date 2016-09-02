# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 15:34:51 2015

@author: sansomk
"""
#
# Write a rectilinear grid to ASCII Tecplot format.
#
def tecplot_WriteRectilinearMesh(filename, X, Y, Z, vars):
    def pad(s, width):
        s2 = s
        while len(s2) < width:
            s2 = ' ' + s2
        if s2[0] != ' ':
            s2 = ' ' + s2
        if len(s2) > width:
            s2 = s2[:width]
        return s2
    def varline(vars, id, fw):
        s = ""
        for v in vars:
            s = s + pad(str(v[1][id]),fw)
        s = s + '\n'
        return s
 
    fw = 10 # field width
 
    f = open(filename, "wt")
 
    f.write('Variables="X","Y"')
    if len(Z) > 0:
        f.write(',"Z"')
    for v in vars:
        f.write(',"%s"' % v[0])
    f.write('\n\n')
 
    f.write('Zone I=' + pad(str(len(X)),6) + ',J=' + pad(str(len(Y)),6))
    if len(Z) > 0:
        f.write(',K=' + pad(str(len(Z)),6))
    f.write(', F=POINT\n')
 
    if len(Z) > 0:
        id = 0
        for k in xrange(len(Z)):
            for j in xrange(len(Y)):
                for i in xrange(len(X)):
                    f.write(pad(str(X[i]),fw) + pad(str(Y[j]),fw) + pad(str(Z[k]),fw))
                    f.write(varline(vars, id, fw))
                    id = id + 1
    else:
        id = 0
        for j in xrange(len(Y)):
            for i in xrange(len(X)):
                f.write(pad(str(X[i]),fw) + pad(str(Y[j]),fw))
                f.write(varline(vars, id, fw))
                id = id + 1
 
    f.close()
    
if __name__ == '__main__':
    
    import math
     
    # Coordinates
    X = (0., 1., 1.5, 2., 2.5, 3., 4.)
    Y = (0., 1, 2., 3., 4.)
    Z = (0., 2., 4.)
     
    # Data
    radius = []
    nodeid = []
    id = 0
    for k in xrange(len(Z)):
        for j in xrange(len(Y)):
            for i in xrange(len(X)):
                r = math.sqrt(X[i]*X[i] + Y[j]*Y[j] + Z[k]*Z[k])
                radius = radius + [r]
                nodeid = nodeid + [id]
                id = id + 1
     
    # Write the data to Tecplot format
    vars = (("radius", radius), ("nodeid", nodeid))
    tecplot_WriteRectilinearMesh("2drect.tec", X, Y, [], vars)  # 2D
    tecplot_WriteRectilinearMesh("3drect.tec", X, Y, Z, vars)   # 3D