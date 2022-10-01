#!/usr/bin/env python3

from cv2 import destroyAllWindows
import annotate
from annotate import cv2
from annotate import np
from utils import MyWarp
import rectifications
import argparse

def get_projective_line(p1 , p2):
    p1 = np.append(np.flip(p1) , 1)
    p2 = np.append(np.flip(p2) , 1)
    line = np.expand_dims(np.cross(p1,p2) ,axis=0)
    print(line)
    return line

def new_color():
    color = np.random.randint(0 , 255 , size=3 , dtype=int)
    return color.tolist()


def make_lines(image_path, points):
    
    lines = np.reshape(np.array([] , dtype=int) , (0,3))
    img = cv2.imread(image_path)
    if(len(points)%2 !=0):
        print("Number of points selected not even")
        return
    for i in range(int(len(points) /2)):
        p1  = points[2*i,:]
        p2 = points[2*i +1,:]
        if (i%2 ==0):
            color = new_color()
        line_img = cv2.line(img , p1, p2 , color , 2)
        lines = np.append(lines , get_projective_line(p1,p2) ,axis =0)

    cv2.imshow("Line" , line_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return lines

def get_points(image_path):
    points = annotate.annotate(image_path)
    print("Points received: \n", points )
    return points

def process_image(args1):
    image_path1 = args1.image_path1
    if(image_path1 == None):
        raise Exception("provided image_path1 is null")
    
    points = get_points(image_path1)
    lines = np.asarray(make_lines(image_path1 , points) , dtype=float)
    print("Lines recieved: \n" , lines)
    lines = lines / np.reshape(lines[:,2] , (-1,1))
    return lines

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Main file for assignment 1 of 16822 Geometry-Based Vision")
    parser.add_argument('--type' , type=str , required=True , choices=['q1' , 'q2' , 'q3'])
    parser.add_argument('--image_path1' , type=str , required=False)
    parser.add_argument('--image_path2' , type=str , required=False)
    parser.add_argument('--output_name' , type=str , required=False)
    
    args = parser.parse_args()
    qtype = args.type

    if(qtype == 'q1'):
        print("Affine rectification")
        lines = process_image(args)
        Ha = rectifications.affine(lines)

        orig_img  = cv2.imread(args.image_path1)
        new_img = MyWarp(orig_img , Ha)
        cv2.imwrite('processed_images/affine/' + args.output_name + '.jpg' , new_img)
        cv2.imshow('Affine' , new_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



    elif(qtype == 'q2'):
        print("Metric rectification")
        lines = process_image(args)
        print(lines)

        Hp = rectifications.metric(lines)
        orig_img_m  = cv2.imread(args.image_path1)
        new_img_m = MyWarp(orig_img_m , Hp)
        cv2.imshow('Metric' , new_img_m)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        

    elif(qtype == 'q3'):
        print("Homography estimation")
        img_path1 = args.image_path1
        img_path2 = args.image_path2
        img1 = cv2.imread(img_path1)
        img2 = cv2.imread(img_path2)

        # pts1 = np.reshape(np.array([] , dtype=float) , (0,2))
        # h,w = img1.shape[:2]
        # pts1 = np.append(pts1 , np.expand_dims(np.asarray([0 ,0]) , axis=0) , axis=0)
        # pts1 = np.append(pts1 , np.expand_dims(np.asarray([0 ,w]) , axis=0) , axis=0)
        # pts1 = np.append(pts1 , np.expand_dims(np.asarray([h ,w]) , axis=0) , axis=0)
        # pts1 = np.append(pts1 , np.expand_dims(np.asarray([h ,0]) , axis=0) , axis=0)
        
        pts1 = get_points(img_path1)
        pts2 = get_points(img_path2)
        
        print(img1.shape)
        print(img2.shape)

        Hh , pts1_homo = rectifications.homography(pts1 , pts2)
        new_img1 = MyWarp(img1 , Hh)
        warped_img1 = np.zeros_like(img2)

        #Check first point in warped image
        nonzero_pts = np.transpose(np.nonzero(new_img1))
        nonzero_pts = nonzero_pts[:, 0:2]
        first_pixel = nonzero_pts[0,:]
        ##

        pts1np = np.fliplr(pts1)
        pts2np = np.fliplr(pts2)
        print(pts2np)

        warped_img1[pts2np[0,0]- first_pixel[0]:pts2np[0,0]+new_img1.shape[0] - first_pixel[0], pts2np[0,1]- first_pixel[1]: pts2np[0,1]+new_img1.shape[1] - first_pixel[1] , :] = new_img1
        warped_nonzero = np.nonzero(warped_img1)
        c1_nonzero = (warped_nonzero[0] , warped_nonzero[1] , np.zeros_like(warped_nonzero[1]))
        c2_nonzero = (warped_nonzero[0] , warped_nonzero[1] , np.ones_like(warped_nonzero[1]))
        c3_nonzero = (warped_nonzero[0] , warped_nonzero[1] , 2*np.ones_like(warped_nonzero[1]))
        
        ##removing pixels in image 2
        img2[c1_nonzero] = 0
        img2[c2_nonzero] = 0
        img2[c3_nonzero] = 0

        combined_img = cv2.add(img2 ,warped_img1 )
        cv2.imshow('combined' , combined_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    else:
        print("No such case")
    
    
    