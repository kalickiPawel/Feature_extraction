from skimage import io
from scipy.interpolate import UnivariateSpline

import numpy as np
import cv2
import os


def find_nearest(array, value):
    dist = []
    array = np.asarray(array)
    value = np.asarray(value)
    for i in range(len(array)):
        dist.append(np.linalg.norm(array[i] - value))
    dist = np.asarray(dist)
    idx = dist.argmin()
    return idx


class SignatureDescriptors:
    dataset_path = ""
    train_set_paths, test_set_paths = [], []
    train_set, test_set = [], []
    results = []

    def __init__(self, train_p, test_p, data_obj):
        self.class_names = sorted(data_obj.class_names)
        self.train_set_paths, self.test_set_paths = train_p, test_p
        self.prepare_img()
        print(f"Number of Train feature: {len(self.train_set)}")
        print(f"Number of Test feature: {len(self.test_set)}")

    def prepare_img(self):
        for train in self.train_set_paths:
            d = {'class_name': os.path.basename(os.path.dirname(train)), 'file': io.imread(train)}
            self.train_set.append(d)
        for test in self.test_set_paths:
            d = {'class_name': os.path.basename(os.path.dirname(test)), 'file': io.imread(test)}
            self.test_set.append(d)

    def fit(self):
        train = [f['file'] for f in self.train_set]
        train_learned = []
        for item in train:
            train_learned.append(self.signature_desc(item))
        default_size = self.find_max_list(train_learned)
        train_learned2 = self.interpolate_to_fixed_size(train_learned, default_size)
        return train_learned2

    @staticmethod
    def signature_desc(img):
        ret, thresh = cv2.threshold(img, 0, 200, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contour = max(contours, key=len)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        centroid = [cX, cY]

        dist = []
        for c in contour:
            dist.append(np.linalg.norm(c - centroid))
        return dist

    @staticmethod
    def find_max_list(given_list):
        list_len = [len(i) for i in given_list]
        return max(list_len)

    @staticmethod
    def interpolate_to_fixed_size(given_list, default_size):
        for i in range(len(given_list)):
            old_indices = np.arange(0, len(given_list[i]))
            new_indices = np.linspace(0, len(given_list[i]) - 1, default_size)
            spl = UnivariateSpline(old_indices, given_list[i], k=1, s=0)
            given_list[i] = spl(new_indices)
        return given_list

    def test(self, clf):
        results = []
        for img in self.test_set:
            results.append(self.classify(img, clf))
        return results

    def classify(self, img, train):
        tmp = self.signature_desc(img['file'])
        default_size = self.find_max_list(train)
        img2 = self.interpolate_to_fixed_size([tmp], default_size)
        res = self.enc_dict(find_nearest(train, img2))
        out = []
        # print("Deskryptor - Sygnatura: ", res)
        out.append(1) if res == img['class_name'] else out.append(0)
        return out

    def enc_dict(self, i):
        return self.class_names[i]

    @staticmethod
    def score(scores):
        res = 0
        for i in range(len(scores)):
            res += scores[i][0]
        res = res / len(scores) * 100
        return res