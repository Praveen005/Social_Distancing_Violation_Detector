# base path to YOLO directory
MODEL_PATH = "yolo-coco"
MODEL_PATH_LAPTOP = "yolo-coco"
MODEL_PATH_WEBCAM = "yolo-coco"

# initialize minimum probability to filter weak detections along with the
# threshold when applying non-maxim suppression
MIN_CONF = 0.3 #last
# MIN_CONF = 0.1
# MIN_CONF = 0.6
NMS_THRESH = 0.3  #last co
# NMS_THRESH = 0.2

# should NVIDIA CUDA GPU be used?
USE_GPU =  True

# define the minimum safe distance (in pixels) that two people can be from each other
MIN_DISTANCE = 50

