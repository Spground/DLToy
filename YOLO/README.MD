# Usage

You can use this YOLO_detect.py to generate a Object Detection Video from origin video.

1. Downloads pre-training model

   download `model_data.tar.gz` file from [https://pan.baidu.com/s/1D4MYVxHdUnege-Kgha1l2A](https://pan.baidu.com/s/1D4MYVxHdUnege-Kgha1l2A), and uncompress that file into `YOLO` dir with `tar -xzvf model_data.tar.gz`.

2. Run python script like `python YOLO_detect.py -r 29.97 -w 1280 -H 720 -f video.mp4 -o .` with your actual arguments such as video rate, height and width.

# Example

1. origin video

<video width="320" height="240" controls>
  <source src="https://github.com/Spground/DLToy/blob/master/YOLO/videoplayback.mp4" type="video/mp4">
</video>

2. detected video

<video width="320" height="240" controls>
  <source src="https://github.com/Spground/DLToy/blob/master/YOLO/videoplayback_detect.mp4" type="video/mp4">
</video>
