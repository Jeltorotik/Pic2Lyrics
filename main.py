import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from google.cloud import vision
from google.cloud.vision import types

from translate import Translator
import re

import io
import os

class Pic2Lyrics():
    
    def __init__(self):
        self.client = vision.ImageAnnotatorClient()
        self.images = []
        self.pointer = 0
        self.translator = Translator(to_lang="ru")
        
    def append_img(self, path):
        self.images.append([-1, cv2.imread(path), None])
        
        
    def video_2_images(self, path, freq = 120):
        vidcap = cv2.VideoCapture(path)
        frame = 0
        success = True
        while success:
            success, image = vidcap.read()
            if frame % freq == 0:
                self.images.append([frame, image, None])
                #cv2.imwrite("frame%d.jpg" % frame, image)     # save frame as JPEG file 
            frame += 1
          
        
    def show_data(self):
        for img in self.images:
            print('Frame: {} \nLabels: {}'.format(img[0], img[2]))
            draw_picture(img[1])
    
    
    def recognition_google(self, image):
        #Takes numpy RGB image
        # Returns list of labels in the image

        image_string = cv2.imencode('.jpg', image)[1].tostring()

        img = types.Image(content = image_string)
        response = self.client.label_detection(image = img)
        labels = response.label_annotations
        output = [label.description for label in labels]

        return output 
    
    
    def image_2_labels(self, rec_sys="google", russian = True):
        
        while self.pointer < len(self.images):
            print("{0}/{1}".format(self.pointer+1, len(self.images)))
            image = self.images[self.pointer]

            if rec_sys == "google":
                
                eng_labels = self.recognition_google(image[1])
                
            elif rec_sys == "yolo":
                pass
                #room for YOLO, or something
                
            # Eng to rus
            if russian:
                labels = []
                for label in eng_labels:
                    rus_label = self.translator.translate(label)
                    #Иногда переводчик оставляет английские слова, поэтому просто удалим их:
                    rus_label = re.sub(r'[^А-Яа-я ]', '', rus_label)
                    if re.sub(r' ', '', rus_label):
                        labels.append(rus_label)
            else:
                labels = eng_labels
                
            self.images[self.pointer][2] = labels
            self.pointer += 1
        print("Complete!")

