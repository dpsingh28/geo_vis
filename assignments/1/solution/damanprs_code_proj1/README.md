# HW1

The basic command for running the code is


    python3 main.py --type <qtype> --option ...


## q1

    python3 main.py --type q1 --image_path1 <path_to_image> --output_name <Desired output name>

- On doing so, an opencv dialog with the image will appear
- Select points for the lines, in an orderly way
- Orderly means, give pt1 and pt2 for first line
- Then give points for the next line
- First select lines for the first paralle set 
- Then go on to slecting points for the next set of parallel lines
- Second dialog box appears with your selected lines
- Third dialog box appears with the rectified image
- Last dialog box appears, for you to select similar lines, for cosine calculation
- cosine angles are printed in the terminal

## q2

    python3 main.py --type q2 --image_path1 <path_to_affine_rectified_image> --output_name <Desired output name>

- The point selection criteria here is similar to the previous question
- Except, the points must be selected for perpendicular set of lines, instead of parallel
- Second dialog box appears with your selected lines
- Third dialog box appears with the rectified image
- Last dialog box appears, for you to select similar lines, for cosine calculation
- cosine angles are printed in the terminal
- 
## q3

    python3 main.py --type q3 --image_path1 <path_to_the_image_to_be_warped> --image_path2 <path_to_reference_image>

In our case

    python3 main.py --type q3 --image_path1 assignment1/data/q3/desk-normal.png --image_path2 assignment1/data/q3/desk-perspective.png

- Here, the point selection must be started from the top left and done in a clockwise manner
- This manner of selection must be same for both the images


### For more information

    python3 main.py --help