# imports
from configs import config
from configs.detection import detect_people
from scipy.spatial import distance as dist 
import numpy as np
import argparse
import imutils
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i1", "--input1", type=str, default="", help="path to (optional) first input video file")
ap.add_argument("-i2", "--input2", type=str, default="", help="path to (optional) second input video file")
ap.add_argument("-o", "--output", type=str, default="", help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1, help="whether or not output frame should be displayed")
args = vars(ap.parse_args())

# load the COCO class labels the YOLO model was trained on
labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

# load the YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# check if GPU is to be used or not
if config.USE_GPU:
    # set CUDA s the preferable backend and target
    print("[INFO] setting preferable backend and target to CUDA...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# determine only the "output" layer names that we need from YOLO
unconnected_layers = net.getUnconnectedOutLayers()
ln = net.getLayerNames()
ln_filtered = []

for i in unconnected_layers:
    
    if isinstance(i, (tuple, list)) and len(i) > 0:
        layer_index = i[0] - 1
        ln_filtered.append(ln[layer_index])

ln = ln_filtered

# initialize the video stream and pointer to output video file
print("[INFO] accessing video stream...")
# open input video if available else webcam stream
vs1 = cv2.VideoCapture(args["input1"] if args["input1"] else 0)
vs2 = cv2.VideoCapture(args["input2"] if args["input2"] else 1)

# if you want to use phone's camera, check for your phone's ip and change the value below accordingly
ip2= 'http://192.168.45.244:8080/video'
ip1= 'http://192.0.0.2:8080/video'

# vs1 = cv2.VideoCapture(ip1)
# vs2 = cv2.VideoCapture(ip2)

# for live feed
# to use laptop's cam, make the argument '0', for external cams use, 1 2 3 ...
# vs1 = cv2.VideoCapture(1)
# vs2 = cv2.VideoCapture(2)

writer = None

# loop over the frames from the video stream
while True:
    # read the next frame from the input video
    (grabbed1, frame1) = vs1.read()
    (grabbed2, frame2) = vs2.read()

    # if the frame was not grabbed, then that's the end fo the stream 
    if not grabbed1 or not grabbed2:
        break

    # resize the frame and then detect people (only people) in it
    frame1 = imutils.resize(frame1, width=700)
    frame2 = imutils.resize(frame2, width=700)
    # results1 = detect_people(frame1, net, ln, personIdx=LABELS.index("bottle"))
    # results2 = detect_people(frame2, net, ln, personIdx=LABELS.index("bottle"))
    
    results1 = detect_people(frame1, net, ln, personIdx=LABELS.index("person"))
    results2 = detect_people(frame2, net, ln, personIdx=LABELS.index("person"))

    # initialize the set of indexes that violate the minimum social distance
    violate = set()
    violate1 = set()

    # ensure there are at least two people detections (required in order to compute the
    # the pairwise distance maps)
    if len(results1) >= 2:
        # extract all centroids from the results and compute the Euclidean distances
        # between all pairs of the centroids
        centroids = np.array([r[2] for r in results1])
        D = dist.cdist(centroids, centroids, metric="euclidean")
        # loop over the upper triangular of the distance matrix
        for i in range(0, D.shape[0]):
            for j in range(i+1, D.shape[1]):
                # check to see if the distance between any two centroid pairs is less
                # than the configured number of pixels
                if D[i, j] < config.MIN_DISTANCE:
                    # update the violation set with the indexes of the centroid pairs
                    violate.add(i)
                    violate.add(j)

                    # for second
    if len(results2) >= 2:
        # extract all centroids from the results and compute the Euclidean distances
        # between all pairs of the centroids
        centroids1 = np.array([r[2] for r in results2])
        D1 = dist.cdist(centroids1, centroids1, metric="euclidean")
        # loop over the upper triangular of the distance matrix
        for i in range(0, D1.shape[0]):
            for j in range(i+1, D1.shape[1]):
                # check to see if the distance between any two centroid pairs is less
                # than the configured number of pixels
                if D1[i, j] < config.MIN_DISTANCE:
                    # update the violation set with the indexes of the centroid pairs
                    violate1.add(i)
                    violate1.add(j)
    
    # loop over the results
    for (i, (prob, bbox, centroid)) in enumerate(results1):
        # extract teh bounding box and centroid coordinates, then initialize the color of the annotation
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid
        color = (0, 255, 0)
        # if the index pair exists within the violation set, then update the color
        if i in violate:
            color = (0, 0, 255)
        # draw (1) a bounding box around the person and (2) the centroid coordinates of the person
        cv2.rectangle(frame1, (startX, startY), (endX, endY), color, 2)
        cv2.circle(frame1, (cX, cY), 5, color, 1)

        # for second
    for (i, (prob, bbox, centroid)) in enumerate(results2):
        # extract teh bounding box and centroid coordinates, then initialize the color of the annotation
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid
        color = (0, 255, 0)
        # if the index pair exists within the violation set, then update the color
        if i in violate1:
            color = (0, 0, 255)
        # draw (1) a bounding box around the person and (2) the centroid coordinates of the person
        cv2.rectangle(frame2, (startX, startY), (endX, endY), color, 2)
        cv2.circle(frame2, (cX, cY), 5, color, 1)

    # draw the total number of social distancing violations on the output frame
    text = "Social Distancing Violations: {}".format(len(violate))
    cv2.putText(frame1, text, (10, frame1.shape[0] - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

    # for second
    text1 = "Social Distancing Violations: {}".format(len(violate1))
    cv2.putText(frame2, text1, (10, frame2.shape[0] - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

    if len(violate) > len(violate1):
        voilation=len(violate1)
    elif len(violate) < len(violate1):
        voilation=len(violate)
    elif (len(results1) == 1 and len(violate1) > 2):
        voilation= len(violate1)
    elif (len(results2) == 1 and len(violate) > 2):
        violation = len(violate)
    else:
        voilation=len(violate)

    combined_frame = np.concatenate((frame1, frame2), axis=1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 4
# Get the text size
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)

# Set the background rectangle position and size
    background_pos = (490, combined_frame.shape[0] - 15 - text_size[1])
    background_size = (text_size[0], text_size[1])
    cv2.rectangle(combined_frame, background_pos, (background_pos[0] + background_size[0], background_pos[1] + background_size[1]), (0, 0, 255) , 4)

    text2 = "Social Distancing Violations after both inputs: {}".format(voilation)
    cv2.putText(combined_frame, text2, (400, combined_frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 4)
    # check to see if the output frame should be displayed to the screen
    if args["display"] > 0:
        # show the output frame
        cv2.imshow("Output", combined_frame)
        key = cv2.waitKey(1) & 0xFF

        # if the 'q' key is pressed, break from the loop
        if key == ord("q"):
            break
    
    # if an output video file path has been supplied and the video writer ahs not been 
    # initialized, do so now
    if args["output"] != "" and writer is None:
        # initialize the video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 25, (combined_frame.shape[1], combined_frame.shape[0]), True)

    # if the video writer is not None, write the frame to the output video file
    if writer is not None:
        print("[INFO] writing stream to output")
        writer.write(combined_frame)

