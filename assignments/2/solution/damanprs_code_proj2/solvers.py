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


    _,_,vh = np.linalg.svd(A)
    P = np.reshape(vh[-1,:] , (3,4))
    # P = P / P[-1,-1]
    return P

def get_K3(vanish_pts):
    v1 = vanish_pts[0,:]
    v2 = vanish_pts[1,:]
    v3 = vanish_pts[2,:]

    A = np.reshape(np.array([] , dtype=float) , (0,4))
    A = np.vstack(( A , np.array([[v1[0]*v2[0]+v1[1]*v2[1] , v1[0]*v2[2] + v1[2]*v2[0] , v1[1]*v2[2] + v1[2]*v2[1] , v1[2]*v2[2] ]]) ))
    A = np.vstack(( A , np.array([[v2[0]*v3[0]+v2[1]*v3[1] , v2[0]*v3[2] + v2[2]*v3[0] , v2[1]*v3[2] + v2[2]*v3[1] , v2[2]*v3[2] ]]) ))
    A = np.vstack(( A , np.array([[v3[0]*v1[0]+v3[1]*v1[1] , v3[0]*v1[2] + v3[2]*v1[0] , v3[1]*v1[2] + v3[2]*v1[1] , v3[2]*v1[2] ]]) ))

    _,_,vh = np.linalg.svd(A)
    w_params = vh[-1,:]
    w_params = w_params / w_params[-1]
    w_params = w_params[0:3]
    a,b,c = w_params
    w_mat = np.array([[a , 0 , b],[0 , a , c],[b , c , 1]])
    print("w matrix:\n",w_mat)
    L = np.linalg.cholesky(w_mat)
    K = np.linalg.inv(L.T)
    K = K / K[-1,-1]
    return K

def get_H(sq_pts , wh_ratio):
    pts2 = sq_pts
    pts1 = np.array([[0,1,1],[wh_ratio*1,1,1],[wh_ratio*1,0,1] , [0,0,1]])
    A = np.reshape(np.array([] , dtype=float) , (0,9))
    for i in range(len(pts1)):
        A = np.append(A , np.expand_dims(np.zeros(9), axis=0) , axis=0)
        A = np.append(A , np.expand_dims(np.zeros(9), axis=0) , axis=0)
        A[2*i,3:6] =  -pts2[i,2]*pts1[i,:]
        A[2*i,6:9] = pts2[i,1]*pts1[i,:]
        A[2*i+1,0:3] = pts2[i,2]*pts1[i,:]
        A[2*i+1,6:9] =  -pts2[i,0]*pts1[i,:]

    
    _,_,vh = np.linalg.svd(A)
    vh = vh / vh[-1,-1]
    H = vh[-1,:]
    H = np.reshape(H , (3,3))
    return H 

def get_K5(H1 , H2, H3):
    A = np.reshape(np.array([] , dtype=float) , (0,6))
    
    def A_sub(H):
        As = np.reshape(np.array([] , dtype=float) , (0,6))
        X = H[:,0]
        Y = H[:,1]
        x1,x2,x3 =X
        y1,y2,y3 =Y
        As = np.vstack(( As , np.array([[x1*y1, x1*y2+x2*y1, x1*y3+x3*y1, x2*y2, x2*y3+x3*y2, x3*y3]])))
        As = np.vstack(( As , np.array([[x1*x1, x1*x2+x2*x1, x1*x3+x3*x1, x2*x2, x2*x3+x3*x2, x3*x3]])
                             -np.array([[y1*y1, y1*y2+y2*y1, y1*y3+y3*y1, y2*y2, y2*y3+y3*y2, y3*y3]])))
        return As

    A = np.vstack((A , A_sub(H1) , A_sub(H2) , A_sub(H3) ))
    _,_,vh = np.linalg.svd(A)
    w = vh[-1,:]
    w1,w2,w3,w4,w5,w6 = w
    w_mat = np.array( [[w1 , w2 , w3] , [w2 , w4 , w5] , [w3 , w5 , w6]] )
    print("W matrix (IAC) is:\n", w_mat)

    L = np.linalg.cholesky(w_mat)
    K = np.linalg.inv(L.T)
    K = K /K[-1,-1]

    return K