# coding: utf-8

import numpy as np
import cv2
import torch
import time
import torchvision
from threading import Thread
from retinaface import RetinaFace

model = torchvision.models.mobilenet_v3_small(pretrained=True)
model.classifier.add_module('Linear', torch.nn.Linear(1000, 1))
model.load_state_dict(torch.load('../model/beauty_model_regressor.pth'))
model.eval()

model_ = torchvision.models.mobilenet_v3_small(pretrained=True)
model_.classifier.add_module('Linear', torch.nn.Linear(1000, 136))
model_.load_state_dict(torch.load('../model/model.pt', map_location = 'cpu'))
model_.eval()
detector = RetinaFace(quality='normal')

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

filters = ['defult','landmarks','evil','fire','Gluttony','storm']
evil = cv2.imread('../filter_image/evil.png', cv2.IMREAD_UNCHANGED)
fire = cv2.imread('../filter_image/fire.png', cv2.IMREAD_UNCHANGED)
Gluttony = cv2.imread('../filter_image/Gluttony.png', cv2.IMREAD_UNCHANGED)
storm = cv2.imread('../filter_image/storm.png', cv2.IMREAD_UNCHANGED)
Elves = cv2.imread('../filter_image/Elves.png', cv2.IMREAD_UNCHANGED)

class FilterStream:
    def __init__(self, filter_idx, image):
        self.filter_idx = filter_idx
        self.image = image
        self.frame = image
        # self.faces = faces
        
        self.stopped = True
        self.t = Thread(target=self.filter, args=())
        self.t.daemon = True
        
    def start(self):
        self.stopped = False
        self.t.start() 

    def filter(self):
        while True :
            if self.stopped is True :
                break
            # self.frame = self.filtering(self.image, self.filter_idx)
            try:
                self.frame = self.filtering(self.image, self.filter_idx)
            except Exception as e:
                print(e)
    
    def filtering(self, image, idx):
        image_copy = np.copy(image)
        
        faces = detector.predict(image)
        if filters[idx]=='defult':
            image_copy = detector.draw(image, faces)
        else:
            for face in faces:
                # x, y, w, h = face['x1'], face['y1'], face['x2']-face['x1'], face['y2']-face['y1']
                x, y, w, h = int(face['x1']-0.4*(face['x2']-face['x1'])), face['y1'], int(1.8*(face['x2']-face['x1'])), face['y2']-face['y1']
                
                resize_image = cv2.resize(image[face['y1']:face['y2'], face['x1']:face['x2']], (256, 256))
                resize_image = transform(resize_image)
                resize_image = resize_image.unsqueeze(0)
                with torch.no_grad():
                    score = model(resize_image)
                    landmarks = model_(resize_image)
                
                landmarks = (landmarks.view(-1, 2) + 0.5).numpy()
                percent_score = (score.item()-1)*25
                
                if filters[idx]=='landmarks':
                    for i, (x, y) in enumerate(landmarks, 1):
                        cv2.circle(image_copy, (int((x * (face['x2']-face['x1'])) + face['x1']), int((y * (face['y2']-face['y1'])) + face['y1'])), 2, [40, 117, 255], -1)
                else:
                    if percent_score>60:
                        resize_filter = cv2.resize(Elves, (w, h))
                    else:
                        if filters[idx]=='evil':
                            resize_filter = cv2.resize(evil, (w, h))
                        elif filters[idx]=='fire':
                            resize_filter = cv2.resize(fire, (w, h))
                        elif filters[idx]=='Gluttony':
                            resize_filter = cv2.resize(Gluttony, (w, h))
                        elif filters[idx]=='storm':
                            resize_filter = cv2.resize(storm, (w, h))
                    roi_color = image[y:y+h, x:x+w]
                    ind = np.argwhere(resize_filter[:,:,3] > 0)
                    for i in range(3):
                        roi_color[ind[:,0],ind[:,1],i] = resize_filter[ind[:,0],ind[:,1],i]
                    image_copy[y:y+h, x:x+w] = roi_color
        return image_copy
    
    def write(self, filter_idx, image):
        self.filter_idx = filter_idx
        self.image = image
        # self.faces = faces
    
    def read(self):
        return self.frame
    
    def stop(self):
        self.stopped = True

class WebcamStream:
    def __init__(self, webcam, delay = 0.03):
        self.delay = delay
        cv2.namedWindow('face detection activated', cv2.WINDOW_KEEPRATIO)
        self.vcap = cv2.VideoCapture(0) if webcam else cv2.VideoCapture('../video/sample_video.mp4')
        # vc.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        
        if self.vcap.isOpened() is False :
            print("[Exiting]: Error accessing webcam stream.")
            exit(0)
        fps_input_stream = int(self.vcap.get(5))
        print('FPS of webcam hardware/input stream: {}'.format(fps_input_stream))
            
        self.grabbed , frame = self.vcap.read()
        self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.grabbed is False :
            print('[Exiting] No more frames to read')
            exit(0)

        self.stopped = True 
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True
        
    def start(self):
        self.stopped = False
        self.t.start() 

    def update(self):
        while True:
            if self.stopped is True :
                break
            
            time.sleep(self.delay)
            self.grabbed , frame = self.vcap.read()
            
            if self.grabbed is False :
                print('[Exiting] No more frames to read')
                self.stopped = True
                break
            self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.vcap.release()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True 

class DetectorStream:
    def __init__(self, image):
        self.image = image
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.faces = detector.predict(image)
        self.stopped = True
        self.t = Thread(target=self.predict, args=())
        self.t.daemon = True
        
    def start(self):
        self.stopped = False
        self.t.start() 

    def predict(self):
        while True :
            if self.stopped is True :
                break
            self.faces = detector.predict(image)
    
    def write(self, image):
        self.image = image
    
    def read(self):
        return self.faces
    
    def stop(self):
        self.stopped = True


if __name__ == '__main__':
    webcam = False
    str = input('Process with webcam? (y/n)')
    if str == 'y':
        webcam = True
    
    delay = 0.01
    filter_idx = 0
    webcam_stream = WebcamStream(webcam, delay = 0.01)
    # detector_stream = DetectorStream(image = webcam_stream.read())
    filter_stream = FilterStream(filter_idx = 0, image = webcam_stream.read())
    # faces = detector_stream.read()
    webcam_stream.start()
    filter_stream.start()
    # detector_stream.start()
    
    while True :
        if webcam_stream.stopped is True:
            break
        
        frame = webcam_stream.read()
        # faces = detector_stream.read()
        # detector_stream.write(frame)
        time.sleep(delay)
        filter_stream.write(filter_idx, frame)
        frame = filter_stream.read()
        
        # 7,13,23,33,37,41,45,51,53,67,69,83,127,131
        frame = cv2.cvtColor(frame, 4)
        cv2.imshow('face detection activated', frame)
        
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('w'):
            filter_idx = len(filters)-1 if filter_idx==0 else filter_idx-1
        elif key & 0xFF == ord('e'):
            filter_idx = 0 if filter_idx==len(filters)-1 else filter_idx+1
        elif key & 0xFF == ord('a'):
            filter_idx = 0
        elif key & 0xFF == ord('s'):
            filter_idx = 1
        elif key & 0xFF == ord('d'):
            filter_idx = 2
        elif key & 0xFF == ord('f'):
            filter_idx = 3
        elif key & 0xFF == ord('g'):
            filter_idx = 4
    webcam_stream.stop()
    filter_stream.stop()
    # detector_stream.stop()
    
    cv2.destroyAllWindows()
