import cv2
import os
import argparse
from tqdm import tqdm
import imageio as iio

parser = argparse.ArgumentParser(description="Generate video from rendered frames")
parser.add_argument("--input_folder", "-i", required=True, type=str, help="Path to the input image folder")
parser.add_argument("--output_file", "-o", required=True, type=str, help="Path to the output video file")
parser.add_argument("--fps", default=20.0, type=float)
parser.add_argument("--start_frame", default=0, type=int)
parser.add_argument("--end_frame", default=400, type=int)
parser.add_argument("--is_render_360", action='store_true', default=True)
args = parser.parse_args()

input_folder = args.input_folder
output_file = args.output_file
fps, start_frame, end_frame = args.fps, args.start_frame, args.end_frame
is_render_360 = args.is_render_360


# Get all image filenames from the input folder and sort them alphabetically
input_files = sorted(os.listdir(input_folder))

if not is_render_360:
    # Read the first image to obtain its size, which will be used to create the video writer
    img = cv2.imread(os.path.join(input_folder, input_files[0], "renders/00000.png"))
    height, width, channels = img.shape

    # Create the video writer for MP4 format
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' is used for MP4 format
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Loop through all image files and write them into the video
    for filename in input_files[start_frame:]:
        img = cv2.imread(os.path.join(input_folder, filename, "renders/00000.png"))
        out.write(img)

else:
    # Read the first image to obtain its size, which will be used to create the video writer
    img = cv2.imread(os.path.join(input_folder, "%05d.png"%start_frame))
    #img = cv2.resize(img, (1236, 822))
    height, width, channels = img.shape

    # Create the video writer for MP4 format
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' is used for MP4 format
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Loop through all image files and write them into the video
    for filename in input_files[start_frame:end_frame]:
        img = cv2.imread(os.path.join(input_folder, filename))
        #img = cv2.resize(img, (1236, 822))
        #print(filename)
        out.write(img)

# Release resources and close the video writer
out.release()
#cv2.destroyAllWindows()


