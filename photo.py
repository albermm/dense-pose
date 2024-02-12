import cv2
from utils.helper import GetLogger, Predictor
from argparse import ArgumentParser
import sys
import matplotlib.pyplot as plt
import numpy as np

parser = ArgumentParser()
parser.add_argument(
    "--input", type=str, help="Set the input path to the video", required=True
)
parser.add_argument(
    "--out", type=str, help="Set the output path to the video", required=True
)
args = parser.parse_args()


logger = GetLogger.logger(__name__)
predictor = Predictor()



if __name__ == "__main__":
    # Example usage for processing a single image
    predictor = Predictor()
    image_path = args.input
    out_frame, out_frame_seg = predictor.predict_single_image(image_path)
    # Now you can do something with the results, such as displaying or saving them


    # Write the frame to the output image
    #output_image_path = args.out
    #cv2.imwrite(output_image_path, out_frame)
    output_image_seg_path = args.out
    cv2.imwrite(output_image_seg_path, out_frame_seg)

