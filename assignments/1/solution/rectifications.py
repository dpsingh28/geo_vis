import numpy as np

def affine(lines):
    p1_inf = np.cross(lines[0,:] ,lines[1,:])
    p2_inf = np.cross(lines[2,:] ,lines[3,:])

    p1_inf = p1_inf / p1_inf[2]
    p2_inf = p2_inf / p2_inf[2]

    print(p1_inf , p2_inf)
    l_inf = np.cross(p1_inf , p2_inf)
    l_inf = l_inf / l_inf[2]
    
    print(l_inf)

    H = np.eye(3)
    H[2,1] = l_inf[0]
    H[2,0] = l_inf[1]

    return H

    

def metric():
    pass