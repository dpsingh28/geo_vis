#!/usr/bin/env python3

import annotate
from annotate import cv2
from annotate import np
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
    points = annotate.annotate(image_path1)
    print("Points received: \n", points )
    return points

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Main file for assignment 1 of 16822 Geometry-Based Vision")
    parser.add_argument('--type' , type=str , required=True , choices=['q1' , 'q2' , 'q3'])
    parser.add_argument('--image_path1' , type=str , required=False)
    parser.add_argument('--image_path2' , type=str , required=False)
    
    args = parser.parse_args()
    qtype = args.type

    if(qtype == 'q1'):
        print("Affine rectification")
        image_path1 = args.image_path1
        if(image_path1 == None):
            raise Exception("provided image_path1 is null")
        points = get_points(image_path1)
        lines = np.asarray(make_lines(image_path1 , points) , dtype=float)
        print("Lines recieved: \n" , lines)
        lines = lines / np.reshape(lines[:,2] , (-1,1))
        print("Lines : \n" , lines)



    elif(qtype == 'q2'):
        print("Affine + Metric rectification")
    elif(qtype == 'q3'):
        print("Homography estimation")
    else:
        print("No such case")
    
    
    