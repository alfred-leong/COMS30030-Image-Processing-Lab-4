################################################
#
# COMS30068 - face.py
# University of Bristol
#
################################################

import numpy as np
import cv2
import os
import sys
import argparse

# LOADING THE IMAGE
# Example usage: python filter2d.py -n car1.png
parser = argparse.ArgumentParser(description='face detection')
parser.add_argument('-name', '-n', type=str, default='images/face1.jpg')
args = parser.parse_args()

# /** Global variables */
cascade_name = "frontalface.xml"


def detectAndDisplay(frame):

	# 1. Prepare Image by turning it into Grayscale and normalising lighting
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    # 2. Perform Viola-Jones Object Detection
    faces = model.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=10, flags=0, minSize=(10,10), maxSize=(300,300))
    # 3. Print number of Faces found
    # print(len(faces))
    # 4. Draw box around faces found
    for i in range(0, len(faces)):
        start_point = (faces[i][0], faces[i][1])
        end_point = (faces[i][0] + faces[i][2], faces[i][1] + faces[i][3])
        colour = (0, 255, 0)
        thickness = 2
        frame = cv2.rectangle(frame, start_point, end_point, colour, thickness)
    return faces


# ************ NEED MODIFICATION ************
def readGroundtruth(filename='groundtruth.txt'):
    predicted_faces = []
    success_count = 0
    valid_count = 0
    # read bounding boxes as ground truth
    with open(filename) as f:
        current_img_name = "face1"
        # read each line in text file
        for line in f.readlines():
            content_list = line.split(",")
            img_name = content_list[0]
            if img_name == current_img_name:
                image = cv2.imread("images/" + img_name + ".jpg", 1)

                current_number = int(img_name[4])
                if current_number - 1 > 0:
                    predicted_count = len(predicted_faces)
                    print("face" + str(current_number - 1))
                    print("success count = " + str(success_count) + " | " + "valid count = " + str(valid_count))
                    print("TPR: " + str(success_count / valid_count))
                    # print("F1: " + str((2 * success_count) / (2 * success_count + (predicted_count - valid_count) + )))
                    print()

                    success_count = 0
                    valid_count = 0

                current_img_name = "face" + str(current_number + 1)
                # find all faces in a picture only once
                predicted_faces = detectAndDisplay(image)
            else:
                image = cv2.imread("images/" + img_name + "_edited.jpg", 1)
            x = float(content_list[1])
            y = float(content_list[2])
            width = float(content_list[3])
            height = float(content_list[4])
            # print(str(x)+' '+str(y)+' '+str(width)+' '+str(height))
            x = int(x)
            y = int(y)
            width = int(width)
            height = int(height)
            start_point = (x, y)
            end_point = (x + width, y + height)
            color = (0, 0, 255) # bounding boxes in red
            thickness = 2
            cv2.rectangle(image, start_point, end_point, color, thickness)
            cv2.imwrite("images/" + img_name + "_edited.jpg", image)
            valid_count += 1
            
            minDiff = float("inf")
            actual_box = [x, y, x + width, y + height] 
            predicted_box = []
            for px, py, pwidth, pheight in predicted_faces:
                if abs(px - x) < minDiff:
                    minDiff = abs(px - x)
                    predicted_box = [px, py, px + pwidth, py + pheight]
            iou = calculateIOU(actual_box, predicted_box)
            THRESHOLD = 0.4
            if iou > THRESHOLD:
                success_count += 1
    
    print("face5")
    print("success count = " + str(success_count) + " | " + "valid count = " + str(valid_count))
    print("TPR: " + str(success_count / valid_count))
    print()

def calculateIOU(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou



# ==== MAIN ==============================================

imageName = args.name

# ignore if no such file is present.
if (not os.path.isfile(imageName)) or (not os.path.isfile(cascade_name)):
    print('No such file')
    sys.exit(1)

# 1. Read Input Image
# frame = cv2.imread(imageName, 1)

# ignore if image is not array.
# if not (type(frame) is np.ndarray):
#     print('Not image data')
#     sys.exit(1)


# 2. Load the Strong Classifier in a structure called `Cascade'
model = cv2.CascadeClassifier()
if not model.load(cv2.samples.findFile(cascade_name)):  # you might need only `if not model.load(cascade_name):` (remove cv2.samples.findFile)
    print('--(!)Error loading cascade model')
    exit(0)


# 3. Detect Faces and Display Result
# detectAndDisplay( frame )

readGroundtruth()

# 4. Save Result Image
# cv2.imwrite( "detected.jpg", frame )


