import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
import time, math


def show_img(window_name , image):
    cv2.imshow(window_name , image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def project_pts(pts):
    pts_new = np.empty((pts.shape[0] , pts.shape[1] +1))
    pts_new[: , 0:pts.shape[1]] = pts
    pts_new[:,-1] = 1
    return pts_new

def normalize_pts(pts):
    center = np.sum(pts , axis=0) / pts.shape[0]
    x0 , y0 = center
    d_avg = pts - center
    d_avg = np.linalg.norm(d_avg , axis=1)
    d_avg = np.sum(d_avg) / len(d_avg)
    sv = math.sqrt(2) / d_avg
    T_mat = np.array([[sv , 0 , -sv*x0] , [0 , sv , -sv*y0] , [0 , 0 , 1]])
    return T_mat

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

def ransac(pts1 , pts2 , points , error_thresh , num_iters):
    start_time = time.time()
    print('Algorithm: ', points, 'point') 
    print('Number of RANSAC iterations: ',num_iters)
    print('Error Threshold: ',error_thresh)
    corres_size = (pts1.copy()).shape[0]
    F_int = None
    F_eight = None
    F_seven = None
    history_inliers = 0
    best_pts1 = None
    best_pts2 = None
    inlier_stack = []
    for i in range(num_iters):
        rand_idx = np.random.randint(0 , pts1.shape[0]-1 , points)
        rand_pts1 = pts1[rand_idx,:]
        rand_pts2 = pts2[rand_idx,:]
        T1 = normalize_pts(rand_pts1)
        T2 = normalize_pts(rand_pts2)
        rand_pts1 = (T1 @ project_pts(rand_pts1).T).T
        rand_pts2 = (T2 @ project_pts(rand_pts2).T).T
        if(points == 8):
            F_eight = T2.T @ eight_pt(rand_pts1 , rand_pts2) @ T1
        else:
            raise RuntimeError("points varible can only be 7 or 8")
        
        idx = None
        inlier_pts1 = None
        inlier_pts2 = None
        def ransac_inliers(F_temp):
            nonlocal idx
            nonlocal inlier_pts1
            nonlocal inlier_pts2
            new_pts2 = np.delete(pts2 , rand_idx , 0)
            new_pts1 = np.delete(pts1 , rand_idx , 0)
            errors = project_pts(new_pts2) * (F_temp @ project_pts(new_pts1).T).T
            errors = np.abs(np.sum(errors , axis=1))
            check = errors < error_thresh
            idx = (np.where(check == True)[0])
            inlier_pts1 = new_pts1[idx , :]
            inlier_pts2 = new_pts2[idx , :]
            n_inliers = idx.shape[0]
            return n_inliers
        
        if points ==8:
            num_inliers = ransac_inliers(F_eight)
            F_int = F_eight

        if num_inliers > history_inliers:
            history_inliers = num_inliers
            best_pts1 = inlier_pts1
            best_pts2 = inlier_pts2
        inlier_stack.append(history_inliers/corres_size)
    
    
    print("Ratio of inliers: ",history_inliers/pts1.shape[0])
    print("Time Taken: ",time.time() - start_time," sec\n")
    new_pt1 = best_pts1
    new_pt2 = best_pts2
    T1 = normalize_pts(new_pt1)
    T2 = normalize_pts(new_pt2)
    new_pt1 = (T1 @ project_pts(new_pt1).T).T
    new_pt2 = (T2 @ project_pts(new_pt2).T).T
    F_new = T2.T @ eight_pt(new_pt1 , new_pt2) @ T1
    F_new = F_new / F_new[-1,-1]
    return F_new

def generate_q1P2(K1 , K2 , corres_pt1 , corres_pt2):
    assert corres_pt1.shape == corres_pt2.shape, "Shapes of the correspondences must be same"
    run = True
    i = 0
    P2_final = None
    while(run):
        i+=1
        print("Iteration ",i," for getting best E")
        Fmat = ransac(corres_pt1 , corres_pt2 , 8 , 0.005 , 10000)
        # print("Fundamental matrix is:\n", Fmat)
        Emat = K2.T @ Fmat @ K1
        # print("Essential matrix is:\n" , Emat)
        u,s,v = np.linalg.svd(Emat)
        s[-1] = 0
        s[0] = (s[0] + s[1])/2
        s[1] = s[0]
        E_mat = u@np.diag(s)@v
        u,_,v = np.linalg.svd(E_mat)
        u3 = np.reshape(u[:,-1] , (-1,1))
        wmat = np.array([[0,-1,0] , [1,0,0] , [0,0,1]])
        P2s = {}
        X2s = {}
        zs = {} 
        try:
            P2s[1] = K2 @ np.hstack((u@wmat@v , u3))
            P2s[2] = K2 @ np.hstack((u@wmat@v , -u3))
            P2s[3] = K2 @ np.hstack((u@wmat.T@v , u3))
            P2s[4] = K2 @ np.hstack((u@wmat.T@v , -u3))
            
            proj_pts2 = project_pts(corres_pt2)
            X2s[1] = (np.linalg.inv(P2s[1].T@P2s[1]))@(P2s[1].T @proj_pts2.T)
            X2s[2] = (np.linalg.inv(P2s[2].T@P2s[2]))@(P2s[2].T @proj_pts2.T)
            X2s[3] = (np.linalg.inv(P2s[3].T@P2s[3]))@(P2s[3].T @proj_pts2.T)
            X2s[4] = (np.linalg.inv(P2s[4].T@P2s[4]))@(P2s[4].T @proj_pts2.T)
        
            zs[1] = np.bincount(X2s[1][-1,:]>0)
            zs[2] = np.bincount(X2s[2][-1,:]>0)
            zs[3] = np.bincount(X2s[3][-1,:]>0)
            zs[4] = np.bincount(X2s[4][-1,:]>0)
        except:
            print("Got singular matrices, going on to next iteration")
            continue
        idx_final = None

        if len(zs) !=0:
            for key,value in zs.items():
                if len(value) == 2:
                    if value[-1] == corres_pt1.shape[0]:
                        print("\n\nFound out best E\n\n")
                        idx_final = key
                        P2_final = P2s[idx_final]
                        run = False
                        break
    print("Final R and t are:\n" , np.linalg.inv(K2)@P2_final)
    print(Emat)
    np.save('./new_data/q1P2_final.npy' , P2_final)

def triangulate(p1 , p2 , cam1 , cam2):
    px1 = np.array([[0 , -p1[2] , p1[1]],
                    [p1[2] , 0 , -p1[0]],
                    [-p1[1] , p1[0] , 0]])
    px2 = np.array([[0 , -p2[2] , p2[1]],
                    [p2[2] , 0 , -p2[0]],
                    [-p2[1] , p2[0] , 0]])
    A1 = px1@cam1
    A2 = px2@cam2
    A_mat = np.zeros((4,4))
    A_mat[0:2,:] = A1[0:2,:]
    A_mat[2:4,:] = A2[0:2,:]
    _,_,vt = np.linalg.svd(A_mat)
    X_vec = np.reshape(vt[-1,:] , (1,-1))
    return X_vec

def triangulation(im1 , im2 , cam1 ,cam2 , pts1 , pts2):
    proj_pts1 = project_pts(pts1)
    proj_pts2 = project_pts(pts2)
    img_3d_pts = np.reshape(np.array([]) , (0,4))
    
    start_time = time.time()
    for p1,p2 in zip(proj_pts1 , proj_pts2):
        img_3d_pts = np.vstack((img_3d_pts , triangulate(p1 , p2 , cam1 , cam2)))

    print("Time taken by loop= ",time.time() - start_time," sec")
    img_3d_pts = img_3d_pts / np.reshape(img_3d_pts[:,-1] , (-1,1))
    img_3d_pts = img_3d_pts[:,0:3]

    fig = plt.figure()
    ax=fig.add_subplot(projection='3d')
    ax.scatter3D(img_3d_pts[:,0] , img_3d_pts[:,1] , img_3d_pts[:,2] , s=10)
    plt.show()



############################################################################### Main Function ###############################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter , description="main file for HW4")
    parser.add_argument('--type' , required=True, choices=['1' , '2' , '3'])
    args = parser.parse_args()

    if(args.type == '1'):
        monument_img1 = cv2.imread('./data/monument/im1.jpg')
        monument_img2 = cv2.imread('./data/monument/im2.jpg')
        K_monument = np.load('./data/monument/intrinsics.npy' , allow_pickle=True)
        K1_monument = K_monument.item()['K1']
        K2_monument = K_monument.item()['K2']
        correspondences = np.load('./data/monument/some_corresp_noisy.npz')
        corres_pt1 = correspondences['pts1']
        corres_pt2 = correspondences['pts2']
        P1 = np.hstack((K1_monument , np.array([[0],[0],[0]])))
        P2 = None
        try:
            P2 = np.load('./new_data/q1P2_final.npy')
        except:
            print("P2 not found, generating P2 for q1")
            generate_q1P2(K1_monument , K2_monument , corres_pt1 , corres_pt2)
            P2 = np.load('./new_data/q1P2_final.npy')
        P2 = P2/P2[-1,-1]
        print("Camera 1: \n", P1)
        print("Camera 2: \n", P2)
        triangulation(monument_img1 , monument_img2 , P1 , P2 , corres_pt1 , corres_pt2)
        