# Facial Keypoint Detection Real-Time Filtering

Built an end-to-end Goblin-mask filtering system. Facial landmark include points around the eyes, nose, and mouth on any face and are used in many applications, from facial tracking to emotion recognition. The completed project takes in any image containing faces and identifies the location of each face and their facial keypoints.

## Quick Start for Windows

1. Download the repository, and navigate to the downloaded folder.

	- Download [**ZIP**]() or git clone this repository
	- unzip Facial-Keypoint-Detection-Real-Time-Filtering-main.zip
	- cd Facial-Keypoint-Detection-Real-Time-Filtering-main

2. Download the datasets.

	- go to [**Facial Beauty Rating Dataset**](https://www.kaggle.com/datasets/pranavchandane/scut-fbp5500-v2-facial-beauty-scores/data)
	- Download and unzip archive.zip under Facial-Keypoint-Detection-Real-Time-Filtering-main/data/
	- go to [**300 Faces In-the-Wild ibug Dataset**](http://dlib.net/files/data/ibug_300W_large_face_landmark_dataset.tar.gz)
	- Download and unzip ibug_300W_large_face_landmark_dataset.tar.gz under Facial-Keypoint-Detection-Real-Time-Filtering-main/data/

3. Install a few required pip packages (including OpenCV).

```
pip install -r requirements.txt
```

4. Train the model by yourself or skip.

```
cd src
python facial_beauty_regressor.py
python face_landmark_detection.py
```

5. Demonstrate the real time filtering with the well-trained model.

```
python demo.py
```

-   press `q` to quit the program
-   press `w` to previous filter
-   press `e` to next filter

