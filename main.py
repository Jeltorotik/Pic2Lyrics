import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from google.cloud import vision
from google.cloud.vision import types

import io
import os

def draw_picture(image, bgr=False):
    b, g, r = cv2.split(image) 
    new_image = cv2.merge([r, g, b])
    plt.figure(figsize=(5, 5))
    plt.axis('off')
    plt.imshow(new_image)
    plt.show()

def preprocess(image, width, height, inter = cv2.INTER_AREA):
    return cv2.resize(image, (width, height), interpolation = inter)

class Pic2Lyrics():
    
    def __init__(self):
        self.client = vision.ImageAnnotatorClient()
        self.images = []
        self.pointer = 0
        
    def video_2_images(self, path, freq = 120):
        vidcap = cv2.VideoCapture(path)
        frame = 0
        success = True
        while success:
            success, image = vidcap.read()
            if frame % freq == 0:
                self.images.append([frame, image, None])
                #cv2.imwrite("frame%d.jpg" % frame, image)     # save frame as JPEG file 
                #print(frame)
                #draw_picture(image)
            frame += 1
    
    def recognition_google(self, image):
        #Takes numpy RGB image
        # Returns list of labels in the image

        image_string = cv2.imencode('.jpg', image)[1].tostring()

        img = types.Image(content = image_string)
        response = self.client.label_detection(image = img)
        labels = response.label_annotations
        output = [label.description for label in labels]

        return output 
    
    def image_2_labels(self, rec_sys="google"):
        
        while self.pointer < len(self.images):
            print("{0}/{1}".format(self.pointer+1, len(self.images)))
            image = self.images[self.pointer]

            if rec_sys == "google":
                
                label = self.recognition_google(image[1])
                
            elif rec_sys == "yolo":
                pass
                #room for YOLO, or something
                
            self.images[self.pointer][2] = label
            self.pointer += 1
