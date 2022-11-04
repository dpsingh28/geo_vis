import numpy as np

def eight_pt(pts1 , pts2):
    assert pts1.shape == pts2.shape , "Shape of point set 1 != Shape of point set 2"
    assert pts1.shape[0] == 8 , "Number of points needed == 8 for eight point algorithm"
    assert pts1.shape[1] == 3 , "Points needed in projective space for eight point algorihtm"
    A_mat = np.reshape(np.array([] , dtype=float) , (0,9))
    for i in range(8):
        u1,v1,w1 = pts1[i,:]
        u2,v2,w2 = pts2[i,:]
        A_mat = np.vstack((A_mat , np.array([[u2*u1 , u2*v1 , u2*w1 , v2*u1 , v2*v1 , v2*w1 , w2*u1 , w2*v1 , w2*w1]]) ))
    
    _,_,vt = np.linalg.svd(A_mat)
    F_mat = np.reshape(vt[-1,:] , (3,3))
    uf,sf,vf = np.linalg.svd(F_mat)
    sf[-1] = 0
    F_sing = uf@np.diag(sf)@vf
    return F_sing