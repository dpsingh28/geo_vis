import numpy as np 
import cv2
import argparse
import math
import time
import annotate
from matplotlib import pyplot as plt
from scipy.optimize import least_squares
from solvers import eight_pt, seven_pt , triangulate

# np.random.seed(5)

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

def getF(object):
    correspondences = np.load('./data/q1a/'+object+'/'+ object +'_corresp_raw.npz')
    pts1 = correspondences['pts1']
    pts2 = correspondences['pts2']
    assert pts1.shape == pts2.shape , "Issue in original data"
    T1 = normalize_pts(pts1)
    T2 = normalize_pts(pts2)
    pts1 = (T1 @ project_pts(pts1).T).T
    pts2 = (T2 @ project_pts(pts2).T).T

    return (T2.T @ eight_pt(pts1 , pts2) @ T1)

def epilines(object1, object2 , F_object):
    object1_pt = annotate.annotate(object1)
    object1_pt = project_pts(object1_pt)
    object2_line = (F_object @ object1_pt.T).T
    line_img = object2.copy()
    for line in object2_line:
        x0, y0 = map(int, [0, -line[2] / line[1] ])
        x1, y1 = map(int, [object2.shape[1], -(line[2] + line[0] * object2.shape[1]) / line[1] ])
        line_img = cv2.line(line_img , (x0 , y0) , (x1 , y1) , (255,0,0) , 2)
    show_img('epipolar line' , line_img)
    cv2.imwrite('./submissions/car2_lines.jpg' , line_img)

def ransac(pts1 , pts2 , object1, object2 , points , error_thresh , num_iters):
    start_time = time.time()
    print('Algorithm: ', points, 'point') 
    print('Number of iterations: ',num_iters)
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
        elif(points==7):
            roots , F_coeff , F_const =  seven_pt(rand_pts1 , rand_pts2) 
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
        elif points == 7:
            real_part_roots = roots.real
            imag_part_roots = roots.imag
            num_real = np.bincount(roots.imag == 0)[-1]
            if(num_real == 1):
                idx_sp = np.where(imag_part_roots == 0)[0]
                F_seven = real_part_roots[idx_sp]*F_coeff + F_const
                num_inliers = ransac_inliers(F_seven)
            elif(num_real==3):
                Fs = np.zeros((3,3,3))
                sevenp_inlners = np.zeros(3)
                Fs[0,:,:] = real_part_roots[0]*F_coeff + F_const
                Fs[1,:,:] = real_part_roots[1]*F_coeff + F_const
                Fs[2,:,:] = real_part_roots[2]*F_coeff + F_const
                for i in range(3):
                    sevenp_inlners[i] = ransac_inliers(Fs[i,:,:])
                bestF_idx = np.argmax(sevenp_inlners)
                F_seven = Fs[bestF_idx , :,:]
                num_inliers = sevenp_inlners[bestF_idx]
            else:
                raise RuntimeError("[Ransac] Error: number of real roots can be only 1 or 3")
            F_int = F_seven

        if num_inliers > history_inliers:
            history_inliers = num_inliers
            best_pts1 = inlier_pts1
            best_pts2 = inlier_pts2
        inlier_stack.append(history_inliers/corres_size)
    
    
    print("Ratio of inliers: ",history_inliers/pts1.shape[0])
    print("Time Taken: ",time.time() - start_time," sec\n")
    plt.plot(inlier_stack)
    plt.xlabel("Number of Iterations")
    plt.ylabel("Ratio of Inliers")
    plt.show()

    # Using the best inliers for fundamental matrix calculation
    new_pt1 = best_pts1
    new_pt2 = best_pts2
    T1 = normalize_pts(new_pt1)
    T2 = normalize_pts(new_pt2)
    new_pt1 = (T1 @ project_pts(new_pt1).T).T
    new_pt2 = (T2 @ project_pts(new_pt2).T).T
    F_new = T2.T @ eight_pt(new_pt1 , new_pt2) @ T1
    F_new = F_new / F_new[-1,-1]
    print("F_best\n:", F_new)
    epilines(object1 , object2 , F_new)


def F_from_7p_roots(roots, F_coeff , F_const , pts1 , pts2):
    real_part_roots = roots.real
    imag_part_roots = roots.imag
    num_real = np.bincount(roots.imag == 0)[-1]

    if(num_real == 1):
        idx = np.where(imag_part_roots == 0)
    elif(num_real==3):
        F0 = real_part_roots[0] * F_coeff + F_const
        F1 = real_part_roots[1] * F_coeff + F_const
        F2 = real_part_roots[2] * F_coeff + F_const
        mag = np.zeros(3)
        mag[0] = np.abs(np.trace(pts2 @ F0 @ pts1.T))
        mag[1] = np.abs(np.trace(pts2 @ F1 @ pts1.T))
        mag[2] = np.abs(np.trace(pts2 @ F2 @ pts1.T))
        idx = np.argmin(mag)
    else:
        raise RuntimeError("Error: number of real roots can be only 1 or 3")

    F_mat = real_part_roots[idx]*F_coeff + F_const
    F_mat = F_mat/F_mat[-1,-1]
    return F_mat

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

    img_3d_color = []
    for i in range(pts1.shape[0]):
        img_3d_color.append( ((im1[pts1[i,1] , pts1[i,0] , :]) /255).tolist())
    img_3d_color = np.array(img_3d_color)
    img_3d_color[:, [0,2]] = img_3d_color[:, [2,0]]

    fig = plt.figure()
    ax=fig.add_subplot(projection='3d')
    ax.set_xlim(-0.8 , 0.8)
    ax.set_ylim(-0.8 , 0.8)
    ax.set_zlim(-0.8 , 0.8)
    ax.w_xaxis.set_pane_color((0, 1, 0, 0.5))
    ax.w_yaxis.set_pane_color((0, 1, 0, 0.5))
    ax.w_zaxis.set_pane_color((0, 1, 0, 0.5))
    ax.scatter3D(img_3d_pts[:,0] , img_3d_pts[:,1] , img_3d_pts[:,2] , s=10 ,color=img_3d_color)
    plt.show()
    return img_3d_pts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description= "Main file for HW3 of 16822")
    parser.add_argument('--type' , required=True, choices=['1A1' , '1A2' , '1B' , '2' , '3' , '4' , '5'])
    parser.add_argument('--ransac_object' , choices=['toybus' , 'toytrain'])
    parser.add_argument('--ransac_algorithm' , choices=['7' ,'8'])
    parser.add_argument('--q5_object' , choices=['car1' , 'car2'])
    args = parser.parse_args()

    if(args.type == '1A1'):
        F_teddy = getF('teddy')
        F_chair = getF('chair')
        print("Fundamental matrix for teddy:\n",F_teddy)
        print("Fundamental matrix for chair:\n",F_chair)
        np.save('./new_data/F_teddy.npy' , F_teddy)
        np.save('./new_data/F_chair.npy' , F_chair)
        object = 'chair'
        object1 = cv2.imread('./data/q1a/'+object+'/image_1.jpg')
        object2 = cv2.imread('./data/q1a/'+object+'/image_2.jpg')
        epilines(object1 , object2 , F_chair)
        object = 'teddy'
        object1 = cv2.imread('./data/q1a/'+object+'/image_1.jpg')
        object2 = cv2.imread('./data/q1a/'+object+'/image_2.jpg')
        epilines(object1 , object2 , F_teddy)
    
    elif(args.type == '1A2'):
        F_teddy = np.load('./new_data/F_teddy.npy')
        F_chair = np.load('./new_data/F_chair.npy')
        teddy_intrinsics = np.load('./data/q1a/teddy/intrinsic_matrices_teddy.npz')
        chair_intrinsics = np.load('./data/q1a/chair/intrinsic_matrices_chair.npz')
        teddy_K1 = teddy_intrinsics['K1']
        teddy_K2 = teddy_intrinsics['K2']
        chair_K1 = chair_intrinsics['K1']
        chair_K2 = chair_intrinsics['K2']
        E_teddy = teddy_K2.T @ F_teddy @ teddy_K1
        E_chair = chair_K2.T @ F_chair @ chair_K1
        print("Essential matrix for teddy\n",E_teddy)
        print("Essential matrix for chair\n",E_chair)

    elif(args.type == '1B'):
        toybus_correspondences = np.load('./data/q1b/toybus/toybus_7_point_corresp.npz')
        toytrain_correspondences = np.load('./data/q1b/toytrain/toytrain_7_point_corresp.npz')
        
        toybus_pts1 = toybus_correspondences['pts1']
        toybus_pts2 = toybus_correspondences['pts2']
        toytrain_pts1 = toytrain_correspondences['pts1']
        toytrain_pts2 = toytrain_correspondences['pts2']

        T1_toybus = normalize_pts(toybus_pts1)
        T2_toybus = normalize_pts(toybus_pts2)
        toybus_pts1 = (T1_toybus @ project_pts(toybus_pts1).T).T
        toybus_pts2 = (T2_toybus @ project_pts(toybus_pts2).T).T
        roots_toybus , F1_toybus , F2_toybus = seven_pt(toybus_pts1 , toybus_pts2)
        F_toybus = F_from_7p_roots(roots_toybus , F1_toybus , F2_toybus , toybus_pts1 , toybus_pts2)
        F_toybus = T2_toybus.T @ F_toybus @ T1_toybus
        print("F_toybus:\n" , F_toybus)
        toybus_im1 = cv2.imread("./data/q1b/toybus/image_1.jpg")
        toybus_im2 = cv2.imread("./data/q1b/toybus/image_2.jpg")
        epilines(toybus_im1 , toybus_im2 , F_toybus)

        T1_toytrain = normalize_pts(toytrain_pts1)
        T2_toytrain = normalize_pts(toytrain_pts2)
        toytrain_pts1 = (T1_toytrain @ project_pts(toytrain_pts1).T).T
        toytrain_pts2 = (T2_toytrain @ project_pts(toytrain_pts2).T).T
        roots_toytrain , F1_toytrain , F2_toytrain = seven_pt(toytrain_pts1 , toytrain_pts2)
        F_toytrain = F_from_7p_roots(roots_toytrain , F1_toytrain , F2_toytrain , toytrain_pts1 , toytrain_pts2)
        F_toytrain = T2_toytrain.T @ F_toytrain @ T1_toytrain
        print("F_toytrain:\n", F_toytrain)
        toytrain_im1 = cv2.imread("./data/q1b/toytrain/image_1.jpg")
        toytrain_im2 = cv2.imread("./data/q1b/toytrain/image_2.jpg")
        epilines(toytrain_im1 , toytrain_im2 , F_toytrain)

    elif(args.type == '2'):
        if ((args.ransac_object == None) or (args.ransac_algorithm == None)):
            raise RuntimeError("arguments ransac_object and ransac_algorithm needed for Q2. Please use python3 main.py -h for more info")
        
        object = args.ransac_object
        raw_correspondences = np.load('./data/q1b/'+object+'/'+object+'_corresp_raw.npz')
        pts1 = raw_correspondences['pts1']
        pts2 = raw_correspondences['pts2']
        object1 = cv2.imread('./data/q1b/'+object+'/image_1.jpg')
        object2 = cv2.imread('./data/q1b/'+object+'/image_2.jpg')
        print("Object: ", object)
        
        if(object == 'toybus'):
            if(args.ransac_algorithm == '8'):
                ransac(pts1 , pts2 , object1 , object2 , 8 , 0.0065 , 10000)
            else:
                ransac(pts1 , pts2 , object1 , object2 , 7 , 440 , 10000)            
        else:
            if(args.ransac_algorithm == '8'):
                ransac(pts1 , pts2 , object1 , object2 , 8 , 0.008 , 10000)
            else:
                ransac(pts1 , pts2 , object1 , object2 , 7 , 650 , 10000)

    elif(args.type == '3'):
        cow1 = cv2.imread('./data/q3/img1.jpg')
        cow2 = cv2.imread('./data/q3/img2.jpg')
        cow_cam1 = np.load('./data/q3/P1.npy')
        cow_cam2 = np.load('./data/q3/P2.npy')
        cow_pts1 = np.load('./data/q3/pts1.npy')
        cow_pts2 = np.load('./data/q3/pts2.npy')
        _ = triangulation(cow1 , cow2 , cow_cam1 ,cow_cam2 , cow_pts1 , cow_pts2)

    elif(args.type == '4'):
        cow1 = cv2.imread('./data/q4/img1.jpg')
        cow2 = cv2.imread('./data/q4/img2.jpg')
        cow_cam1 = np.load('./data/q4/P1_noisy.npy')
        cow_cam2 = np.load('./data/q4/P2_noisy.npy')
        cow_pts1 = np.load('./data/q4/pts1.npy')
        cow_pts2 = np.load('./data/q4/pts2.npy')
        
        proj_pts1 = project_pts(cow_pts1)
        proj_pts2 = project_pts(cow_pts2)
        
        def residual_func(params):
            P1 = params[:12].reshape(3,4)
            P2 = params[12:24].reshape(3,4)
            X3d = params[24:].reshape(-1,3)
            X3d_proj = project_pts(X3d)
            x1_hat = (P1 @ X3d_proj.T).T
            x2_hat = (P2 @ X3d_proj.T).T
            residual  = np.square((proj_pts1[:,0]/proj_pts1[:,2]) - (x1_hat[:,0]/x1_hat[:,2])) 
            residual += np.square((proj_pts1[:,1]/proj_pts1[:,2]) - (x1_hat[:,1]/x1_hat[:,2])) 
            residual += np.square((proj_pts2[:,0]/proj_pts2[:,2]) - (x2_hat[:,0]/x2_hat[:,2])) 
            residual += np.square((proj_pts2[:,1]/proj_pts2[:,2]) - (x2_hat[:,1]/x2_hat[:,2]))
            return residual

        pts3d_init = triangulation(cow1 , cow2 , cow_cam1 ,cow_cam2 , cow_pts1 , cow_pts2)
        params_init = np.zeros(((2*12)+(3*cow_pts1.shape[0])))
        params_init[:12] = cow_cam1.flatten()
        params_init[12:24] =  cow_cam2.flatten()
        params_init[24:] = pts3d_init.flatten()
        start_time = time.time()
        res = least_squares(residual_func , params_init , verbose=2, method='dogbox')
        print("Time taken for optimization: ", time.time() - start_time,' sec')

        final_params = res.x
        P1_final = final_params[:12].reshape(3,4)
        P2_final = final_params[12:24].reshape(3,4)
        X3d_final = final_params[24:].reshape(-1,3)

        print("P1_final:\n", P1_final)
        print("P2_final:\n", P2_final)
        # print(X3d_final.shape)
        _ = triangulation(cow1 , cow2 , P1_final ,P2_final , cow_pts1 , cow_pts2)
    
    elif(args.type == '5'):
        if(args.q5_object == None):
            raise RuntimeError('q5_object argument needed')

        def extract_keyp_and_ransac(object):
            img1 = cv2.imread('./new_data/'+object+'1.jpg')
            img2 = cv2.imread('./new_data/'+object+'2.jpg')
            gray_img1 = cv2.cvtColor(img1 , cv2.COLOR_BGR2GRAY)
            gray_img2 = cv2.cvtColor(img2 , cv2.COLOR_BGR2GRAY)

            sift = cv2.SIFT_create(nfeatures=500)
            kp1 ,desc1 = sift.detectAndCompute(gray_img1 , None)
            kp2 ,desc2 = sift.detectAndCompute(gray_img2 , None)

            bf = cv2.BFMatcher()
            matches = bf.knnMatch(desc1 , desc2 , k=2)
            good = []
            for m,n in matches:
                if m.distance < 0.8*n.distance:
                    good.append([m])
        
            matched_img = cv2.drawMatchesKnn(gray_img1, kp1, gray_img2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            show_img('matches' , matched_img)
            cv2.imwrite('./submissions/car2_matches.jpg' , matched_img)
            pts1_list = np.array([kp1[(mat[0]).queryIdx].pt for mat in good] , dtype=int)
            pts2_list = np.array([kp2[(mat[0]).trainIdx].pt for mat in good] , dtype=int)
        
            print("Ransac for image")
            ransac(pts1_list , pts2_list , img1 , img2 , 8 , 0.1 , 10000)
        
        extract_keyp_and_ransac(args.q5_object)