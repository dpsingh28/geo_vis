import numpy as np

def eight_pt(pts1 , pts2):
    assert pts1.shape == pts2.shape , "Shape of point set 1 != Shape of point set 2"
    assert pts1.shape[1] == 3 , "Points needed in projective space for eight point algorihtm"

    A_mat = np.reshape(np.array([] , dtype=float) , (0,9))
    for i in range(pts1.shape[0]):
        u1,v1,w1 = pts1[i,:]
        u2,v2,w2 = pts2[i,:]
        A_mat = np.vstack((A_mat , np.array([[u2*u1 , u2*v1 , u2*w1 , v2*u1 , v2*v1 , v2*w1 , w2*u1 , w2*v1 , w2*w1]]) ))
    
    _,_,vt = np.linalg.svd(A_mat)
    F_mat = np.reshape(vt[-1,:] , (3,3))
    uf,sf,vf = np.linalg.svd(F_mat)
    sf[-1] = 0
    F_sing = uf@np.diag(sf)@vf
    return F_sing

def seven_pt(pts1 , pts2):
    assert pts1.shape == pts2.shape , "Shape of point set 1 != Shape of point set 2"
    assert pts1.shape[1] == 3 , "Points needed in projective space for seven point algorihtm"
    assert pts1.shape[0]==7, "Number of corespondences should be exactly 7 for sevn point algorithm"

    A_mat = np.reshape(np.array([] , dtype=float) , (0,9))
    for i in range(pts1.shape[0]):
        u1,v1,w1 = pts1[i,:]
        u2,v2,w2 = pts2[i,:]
        A_mat = np.vstack((A_mat , np.array([[u2*u1 , u2*v1 , u2*w1 , v2*u1 , v2*v1 , v2*w1 , w2*u1 , w2*v1 , w2*w1]]) ))
    
    _,_,vt = np.linalg.svd(A_mat)
    
    F1 = np.reshape(vt[-1,:] , (3,3))
    F2 = np.reshape(vt[-2,:] , (3,3))
    F_coeff = F1 - F2
    F_const = F2
    dets = np.zeros(8)
    dets[0] = np.linalg.det(F_coeff)
    dets[1] = np.linalg.det(np.hstack(( np.reshape(F_coeff[:,0] , (-1,1)) , np.reshape(F_coeff[:,1] , (-1,1)) , np.reshape(F_const[:,2] , (-1,1)) )))
    dets[2] = np.linalg.det(np.hstack(( np.reshape(F_coeff[:,0] , (-1,1)) , np.reshape(F_const[:,1] , (-1,1)) , np.reshape(F_coeff[:,2] , (-1,1)) )))
    dets[3] = np.linalg.det(np.hstack(( np.reshape(F_const[:,0] , (-1,1)) , np.reshape(F_coeff[:,1] , (-1,1)) , np.reshape(F_coeff[:,2] , (-1,1)) )))
    dets[4] = np.linalg.det(np.hstack(( np.reshape(F_coeff[:,0] , (-1,1)) , np.reshape(F_const[:,1] , (-1,1)) , np.reshape(F_const[:,2] , (-1,1)) )))
    dets[5] = np.linalg.det(np.hstack(( np.reshape(F_const[:,0] , (-1,1)) , np.reshape(F_coeff[:,1] , (-1,1)) , np.reshape(F_const[:,2] , (-1,1)) )))
    dets[6] = np.linalg.det(np.hstack(( np.reshape(F_const[:,0] , (-1,1)) , np.reshape(F_const[:,1] , (-1,1)) , np.reshape(F_coeff[:,2] , (-1,1)) )))
    dets[7] = np.linalg.det(F_const)

    coeffs = np.zeros(4)
    coeffs[0] = dets[0]
    coeffs[1] = dets[1] + dets[2] + dets[3]
    coeffs[2] = dets[4] + dets[5] + dets[6]
    coeffs[3] = dets[7]

    roots = np.roots(coeffs)
    return roots , F_coeff , F_const