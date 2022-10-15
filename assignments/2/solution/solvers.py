import numpy as np
from regex import P

def get_cam_matrix(im_pts, world_pts):
    assert im_pts.shape[0] == world_pts.shape[0] , "Number of 2D and 3D points are not equal"

    A = np.reshape(np.array([] , dtype=float) , (0,12))
    for i in range(im_pts.shape[0]):
        x = im_pts[i,0]
        y = im_pts[i,1]
        w = im_pts[i,2]

        A = np.append(A , np.zeros((1,12)) , axis=0)
        A = np.append(A , np.zeros((1,12)) , axis=0)
        A[2*i , 4:8] = -w*world_pts[i]
        A[2*i , 8:12] = y*world_pts[i]
        A[2*i +1 , 0:4] = w*world_pts[i]
        A[2*i+1 , 8:12] = -x*world_pts[i]


    u,s,vh = np.linalg.svd(A)
    P = np.reshape(vh[-1,:] , (3,4))
    # P = P / P[-1,-1]
    return P

def get_w3(vanish_pts):
    v1 = vanish_pts[0,:]
    v2 = vanish_pts[1,:]
    v3 = vanish_pts[2,:]

    A = np.reshape(np.array([] , dtype=float) , (0,4))
    A = np.vstack(( A , np.array([[v1[0]*v2[0]+v1[1]*v2[1] , v1[0]*v2[2] , v1[1]*v2[2] , v1[2]*v2[2]]]) ))
    A = np.vstack(( A , np.array([[v2[0]*v3[0]+v2[1]*v3[1] , v2[0]*v3[2] , v2[1]*v3[2] , v2[2]*v3[2]]]) ))
    A = np.vstack(( A , np.array([[v3[0]*v1[0]+v3[1]*v1[1] , v3[0]*v1[2] , v3[1]*v1[2] , v3[2]*v1[2]]]) ))

    u,s,vh = np.linalg.svd(A)
    w_params = vh[-1,:]
    w_params = w_params / w_params[-1]
    w_params = w_params[0:3]
    a,b,c = w_params

    w_mat = np.eye(3)
    w_mat[0,0] = a
    w_mat[1,1] = a
    w_mat[0,2] = b
    w_mat[1,2] = c

    return w_mat