import cv2
import numpy as np
import argparse


def show_img(window_name , image):
    cv2.imshow(window_name , image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()






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