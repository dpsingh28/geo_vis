from matplotlib.pyplot import axis
import numpy as np

def affine(lines):
    p1_inf = np.cross(lines[0,:] ,lines[1,:])
    p2_inf = np.cross(lines[2,:] ,lines[3,:])

    p1_inf = p1_inf / p1_inf[2]
    p2_inf = p2_inf / p2_inf[2]

    print(p1_inf , p2_inf)
    
    l_inf = np.cross(p1_inf , p2_inf)
    l_inf = l_inf / l_inf[2]    

    H = np.eye(3)
    H[2,1] = l_inf[0] 
    H[2,0] = l_inf[1] 

    return H

    

def metric(lines):
    num = len(lines)
    if(num%2 != 0): 
        num-=1
    L = np.reshape(np.array([] , dtype=float) , (0,3))

    for i in range(int(num/2)):
        l1 = lines[2*i, :]
        m1 = lines[2*i +1, :]
        L = np.append(L , np.expand_dims(np.array([m1[0]*l1[0] , m1[1]*l1[0] + m1[0]*l1[1] , m1[1]*l1[1]]) , axis=0) , axis =0)
    
    u,s,vh = np.linalg.svd(L)
    vh = vh / vh[-1,-1]
    a,b,c = vh[:, -1]

    C = np.zeros((3,3))
    C[0,0] = a
    C[0,1] = b
    C[1,0] = b
    C[1,1] = c

    u,s,vh = np.linalg.svd(C)

    S = np.eye(3)
    S[0,0] = np.sqrt(1/s[0])
    S[1,1] = np.sqrt(1/s[1])

    return S@(u.T)


def homography(pts1, pts2):
    ones = np.reshape(np.ones(len(pts1)) ,(-1,1))
    pts1 = np.append(pts1 , ones , axis=1)
    pts2 = np.append(pts2 , ones , axis=1)

    print(pts1)
    print(pts2)
    
    A = np.reshape(np.array([] , dtype=float) , (0,9))
    for i in range(len(pts1)):
        A = np.append(A , np.expand_dims(np.zeros(9), axis=0) , axis=0)
        A = np.append(A , np.expand_dims(np.zeros(9), axis=0) , axis=0)
        A[2*i+1,3:6] = -pts2[i,2]*pts1[i,:]
        A[2*i+1,6:9] =  pts2[i,1]*pts1[i,:]
        A[2*i,0:3] =  -pts2[i,2]*pts1[i,:]
        A[2*i,6:9] = pts2[i,0]*pts1[i,:]
    
    print(A)

    u,s,vh = np.linalg.svd(A)

    print(vh)
    print(s)
    vh = vh / vh[-1,-1]
    
    H = vh[-1,:]
    print(H)
    H = np.reshape(H , (3,3))
    print(H)

    return H , pts1