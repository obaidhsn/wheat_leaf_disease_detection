import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.models import load_model

class WheatDiseasePredictor:
    def __init__(self, csv_path, model_path):
        self.csv_path = csv_path
        self.model_path = model_path
        self.model = self.load_model()
        self.img_size, self.scale = self.load_class_info()

    def load_model(self):
        print('Loading model...')
        return load_model(self.model_path)

    def load_class_info(self):
        class_df = pd.read_csv(self.csv_path)
        img_height = int(class_df['height'].iloc[0])
        img_width = int(class_df['width'].iloc[0])
        img_size = (img_width, img_height)
        scale = class_df['scale by'].iloc[0]
        try:
            s = int(scale)
            s1, s2 = 0, 1
        except:
            split = scale.split('-')
            s1 = float(split[1])
            s2 = float(split[0].split('*')[1])
        return img_size, (s1, s2)

    def predict(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            return None, None

        img = cv2.resize(img, self.img_size)
        img = img * self.scale[1] - self.scale[0]
        img = np.expand_dims(img, axis=0)
        p = np.squeeze(self.model.predict(img))
        index = np.argmax(p)
        prob = p[index]
        class_df = pd.read_csv(self.csv_path)
        class_name = class_df['class'].iloc[index]
        return class_name, prob
