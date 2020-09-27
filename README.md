# TGI-ObjectDetection

## Steps
1. Install Python 3.8
2. pip3 install tensorflow==1.13.1
3. pip3 install opencv-python
4. pip3 install keras==2.2.4
5. pip3 install numpy==1.16.1
6. pip3 install imageai --upgrade

## Downloads
Clone this repository

## ObjectDetection_ssd_v3_coco
1. Follow the run steps
## ObjectDetection_Yolo_h5
1. Download yolo.h5 (https://github.com/OlafenwaMoses/ImageAI/releases/tag/1.0/)
2. Move to folder
3. Follow the run steps

## ObjectDetection_Yolo_v3
1. Download YOLOv3-320 WEIGHTS (https://pjreddie.com/darknet/yolo/)
2. After download, rename the file to "yolov3.weights"
3. Move the file to "cfg"
4. Follow the run steps


## Run Macbook
1. Choose one and open the folder
2. Open in terminal the folder and execute script:
3. export PATH="/usr/local/opt/python/libexec/bin:$PATH"
4. pip install virtualenv virtualenvwrapper 
5. export VIRTUALENVWRAPPER_PYTHON=/usr/local/bin/python 
6. export WORKON_HOME=$HOME/.virtualenvs
7. source /usr/local/share/python/virtualenvwrapper.sh
8. python main.py
