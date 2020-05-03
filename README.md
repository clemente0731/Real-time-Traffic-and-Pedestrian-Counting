# Real-time-Traffic-and-Pedestrian-Counting

<a name="0Zy34"></a>
# Introduction
This project focuses " counting and statistics of moving targets we care about ", drive by YOLOv3 which was Implemented in Tensorflow2."<br />It needs to be stated that the YOLOv3 detector of this project is forked from the nice implementation of [YunYang1994](https://github.com/YunYang1994/TensorFlow2.0-Examples/tree/master/4-Object_Detection/YOLOV3)
<a name="HbIRq"></a>
# Project Demo

- The demo is available on Youtube and Bilibili
- on my laptop gtx1060 FPS reached 15-20
<a name="fOSw7"></a>
# Installation
Reproduce the environment
```
 conda env create -f environment.yml
 wget https://pjreddie.com/media/files/yolov3.weights
```
two test videos are prepared [here](https://drive.google.com/drive/folders/16ZYObAm48Y0ImnCjtUIzeasyp2QaPphI?usp=sharing), you should download.<br />

<a name="qyyHA"></a>
# Parameter adjustment

- For video_demo.py
  -  `video_path = "./vehicle.mp4"`
  - `num_classes = 80`
  - `utils.load_weights(model, "./yolov3.weights")`
- For utils.py
  - `specified_class_id_filter = 2`
  - `line = [(0, 530), (2100, 530)]`
<a name="2YXn3"></a>
# Run demo:
```
conda activate your_env_name
python video_demo.py
```


<a name="DlQMB"></a>
# Citation
If you use this code for your publications, please cite it as:
```
@ONLINE{vdtc,
    author = "Clemente420",
    title  = "Real-time-Traffic-and-Pedestrian-Counting",
    year   = "2020",
    url    = "https://github.com/Clemente420/Real-time-Traffic-and-Pedestrian-Counting"
}
```
<a name="b8tek"></a>
# Author

- Please contact for dataset or more info: clemente0620@gmail.com
<a name="3Lk78"></a>
# License
This system is available under the MIT license. See the LICENSE file for more info.

