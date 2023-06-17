# Social Distancing Violation Detector

<br>

## Installation of the Project:package:

1. Clone the repo

```bash
   $ git clone https://github.com/Praveen005/Social_Distancing_Violation_Detector.git
   $ cd Social_Distancing_Violation_Detector
   $ cd Two_Feeds
```

2. Install dependencies

```bash
   $ pip install -r requirements.txt
```

3. Download the [yolov3.weights](https://github.com/patrick013/Object-Detection---Yolov3/blame/master/model/yolov3.weights) and include it inside the yolo-coco folder
   


4. Two Feeds: Run the main social distancing detector file. (set display to 1 if you want to see output video as processing occurs)
```bash   
   $ python social_distancing_detector.py --input1 pedestrians.mp4 --input2 pedestrians.mp4 --output output.avi --display 1
```
[Caution: The output video stream will not be accurate if you display as processing occurs]

