# Object Pose Annotator

<img src="./lib/object_pose_annotator.png" height="400">


# TODO:
- [X] Show object coordinate
- [X] Add progressive bar
- [X] Add smart view controller
- [X] Move to initial viewpoint
- [X] Add transparency
- [X] Show original image
- [X] Fix camera viewpoint
- [X] Show mask image
- [X] Add score visualizer
- [X] Add world coordinate
- [X] Show object model images
- [X] Visualize the key direction
- [X] Add instance label
- [X] Add logger
- [X] Test on window
- [X] Cx_freeze


## Install and Run

In windows, Microsoft Visual C++ 14.0 is required. [link](https://www.microsoft.com/ko-KR/download/details.aspx?id=48159) [link](https://visualstudio.microsoft.com/ko/visual-cpp-build-tools/)

```
$ conda create -n gc6d-anno python=3.7
$ conda activate gc6d-anno
$ pip install -r requirements.txt
$ python object_pose_annotator.py
```


## 

- `W, A, S, D`: up, left, down, right
- `Q, W`: zoom-

w a s d
q