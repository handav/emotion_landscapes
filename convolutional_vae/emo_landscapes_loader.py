from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import os.path

class EmotionLoader(object):

    """Data Handler that loads robot pushing data."""

    def __init__(self, transform, all_labels=False): #image_width, crop_sz=300):
        #self.crop_sz = 300
        #self.image_width = image_width
        self.transform = transform
        self.all_labels = all_labels
        fname = '../cleaned.csv'
        emotions = {
                'joy'           : 0,
                'anticipation'  : 1,
                'fear'          : 2,
                'sadness'       : 3,
                'surprise'      : 4,
                'trust'         : 5,
                'disgust'       : 6,
                'anger'         : 7,
                'none'          : 8}


        data = []
        for line in open(fname, 'r'):
            path, labels = line.split(',')
            path = '.' + path
            labels = [emotions[l] for l in labels.strip().split(' ')]
            data.append([path, labels])
        self.data = data
        self.N = len(data)

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        while True: # keep looping until file exists
            # sample randomly from data list
            idx = np.random.randint(self.N)
            fname, labels = self.data[idx]
            if os.path.exists(fname): break
        img = Image.open(fname)
        img = self.transform(img)
        label = labels[np.random.randint(len(labels))]
        if self.all_labels:
            d = torch.zeros(9)
            for l in labels: d[l] += 1 
            d.div_(d.sum())
            return img, label, d
        else:
            return img, label 

