# -*- coding : utf-8-*-
import numpy as np
import cv2
import torch
import time
import torchvision
from threading import Thread
# from retinaface import RetinaFace
from utils import *

torch.set_grad_enabled(False)
cfg = cfg_mnet
net = RetinaFace(cfg=cfg, phase = 'test')
pretrained_dict = torch.load('../model/mobilenet0.25_Final.pth', map_location=lambda storage, loc: storage)
f = lambda x: x.split('module.', 1)[-1] if x.startswith('module.') else x
pretrained_dict = {f(key): value for key, value in pretrained_dict.items()}
net.load_state_dict(pretrained_dict, strict=False)
net.eval()

model = torchvision.models.mobilenet_v3_small(pretrained=True)
model.classifier.add_module('Linear', torch.nn.Linear(1000, 1))
model.load_state_dict(torch.load('../model/beauty_model_regressor.pth'))
model.eval()

model_ = torchvision.models.mobilenet_v3_small(pretrained=True)
model_.classifier.add_module('Linear', torch.nn.Linear(1000, 136))
model_.load_state_dict(torch.load('../model/model.pt', map_location = 'cpu'))
model_.eval()
# detector = RetinaFace(quality='normal')

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

filters = ['defult','landmarks','evil','fire','Gluttony','storm','random']
evil = cv2.imread('../filter_image/evil.png', cv2.IMREAD_UNCHANGED)
fire = cv2.imread('../filter_image/fire.png', cv2.IMREAD_UNCHANGED)
Gluttony = cv2.imread('../filter_image/Gluttony.png', cv2.IMREAD_UNCHANGED)
storm = cv2.imread('../filter_image/storm.png', cv2.IMREAD_UNCHANGED)
Elves = cv2.imread('../filter_image/Elves.png', cv2.IMREAD_UNCHANGED)

def predict(image):
    img = np.copy(image)
    img = np.float32(img)
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    loc, conf, landms = net(img)
    
    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2]])
    landms = landms * scale1
    landms = landms.cpu().numpy()

    inds = np.where(scores > 0.02)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    order = scores.argsort()[::-1]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, 0.4)
    dets = dets[keep, :]
    landms = landms[keep]
    
    dets = np.concatenate((dets, landms), axis=1)
    return dets

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
                self.frame = self.image
                print(e)
    
    def filtering(self, image, idx):
        image_copy = np.copy(image)
        im_height, im_width, _ = image.shape
        
        faces = predict(image)
        # faces = detector.predict(image)
        for i, face in enumerate(faces):
            if face[4] < 0.4:
                continue
            b = list(map(int, face))
            if filters[idx]=='defult':
                cv2.rectangle(image_copy, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cv2.circle(image_copy, (b[5], b[6]), 1, (0, 0, 255), 4)
                cv2.circle(image_copy, (b[7], b[8]), 1, (0, 255, 255), 4)
                cv2.circle(image_copy, (b[9], b[10]), 1, (255, 0, 255), 4)
                cv2.circle(image_copy, (b[11], b[12]), 1, (0, 255, 0), 4)
                cv2.circle(image_copy, (b[13], b[14]), 1, (255, 0, 0), 4)
            # image_copy = detector.draw(image, faces)
            else:
                # x, y, w, h = face['x1'], face['y1'], face['x2']-face['x1'], face['y2']-face['y1']
                # x, y, w, h = int(face['x1']-0.5*(face['x2']-face['x1'])), int(face['y1']-0.3*(face['y2']-face['y1'])), int(2.0*(face['x2']-face['x1'])), int(1.3*(face['y2']-face['y1']))
                x, y, w, h = b[0], b[1], b[2]-b[0], b[3]-b[1]
                try:
                    resize_image = cv2.resize(image[b[1]:b[3], b[0]:b[2]], (256, 256))
                except:
                    continue
                resize_image = transform(resize_image)
                resize_image = resize_image.unsqueeze(0)
                with torch.no_grad():
                    score = model(resize_image)
                    landmarks = model_(resize_image)
                
                landmarks = (landmarks.view(-1, 2) + 0.5).numpy()
                percent_score = (score.item()-1)*25
                if filters[idx]=='landmarks':
                    for x_ratio, y_ratio in landmarks:
                        cv2.circle(image_copy, (int((x_ratio * w) + x), int((y_ratio * h) + y)), 2, [40, 117, 255], -1)
                else:
                    left, right, top, down = max(0, int(x-0.7*w)), min(im_width, int(x+1.7*w)), max(0, int(y-0.3*h)), min(im_height, int(y+1.1*h))
                    x, y, w, h = int(x-0.7*w), int(y-0.3*h), int(2.4*w), int(1.4*h)
                    
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
                        elif filters[idx]=='random':
                            if i%4==0:
                                resize_filter = cv2.resize(evil, (w, h))
                            elif i%4==1:
                                resize_filter = cv2.resize(fire, (w, h))
                            elif i%4==2:
                                resize_filter = cv2.resize(Gluttony, (w, h))
                            elif i%4==3:
                                resize_filter = cv2.resize(storm, (w, h))
                    roi_color = image[top:down, left:right]
                    roi_resize_filter = resize_filter[top-y:down-y, left-x:right-x]
                    ind = np.argwhere(roi_resize_filter[:,:,3] > 0)
                    for i in range(3):
                        roi_color[ind[:,0],ind[:,1],i] = roi_resize_filter[ind[:,0],ind[:,1],i]
                    image_copy[top:down, left:right] = roi_color
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
        fps_input_stream = int(self.vcap.get(5))
        print('FPS of webcam hardware/input stream: {}'.format(fps_input_stream))
            
        self.grabbed , frame = self.vcap.read()
        self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.grabbed is False :
            print('[Exiting] No more frames to read')

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
    filter_stream = FilterStream(filter_idx = filter_idx, image = webcam_stream.read())
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
        elif key & 0xFF == ord('h'):
            filter_idx = 5
        elif key & 0xFF == ord('j'):
            filter_idx = 6
    webcam_stream.stop()
    filter_stream.stop()
    # detector_stream.stop()
    
    cv2.destroyAllWindows()
