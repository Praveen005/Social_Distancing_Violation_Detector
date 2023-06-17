# Social Distancing Violation Detector :mag_right:

<br>

To check if or not people are following social distancing norms using OpenCV and YoloV3


## How it works :hammer_and_wrench:

* Utilizing the YOLO model trained on COCO dataset, we perform object detection specifically to identify individuals within the video stream.

* The system calculates the distances between bounding boxes formed around the detected indivisuals

* By comparing these computed distances with the preset threshold value(50px here), we determine whether or not there is a violation of the social distancing guidelines.

* The current code is compatible to take two different inputs(be it live feed or pre-recorded video) and process them independently for the violations. 

* when you use live cams, you can take two different camera angles to get two different perspective to tackle following three cases:
   * One feed says, there is violation but other says, no violation(can happen when in projected image they are close, this is why another camera feed is used to verify if they are actually close)
   * Both says "No Violation"
   * Both says "Violation"

* To be able to work on live feed, we can change the argument of 'VideoCapture' class accordingly.

## Libraries Used :toolbox:

<li> Python </li>
<li> Open CV </li>
<li> Yolo V3</li>
<li> dnn Module </li>
<li> SciPy </li>
<li> imutils</li>
<li> Argparse </li>
<li> Numpy </li>

<br>

## Installation of the Project :package:

Check Inside