import annotate
import argparse


def make_lines():
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path' , type=str , required=True)
    args = parser.parse_args()

    image_path = args.image_path
    points = annotate.annotate(image_path)

    print("Points received: ", points )