from skimage import io

import numpy as np
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


class FourierDescriptor:
    dataset_path = ""
    train_set_paths, test_set_paths = [], []
    train_set, test_set = [], []
    results = []

    def __init__(self, train_p, test_p, data_obj, fft_sizes):
        self.fft_sizes = fft_sizes
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

    def fit(self, size):
        train = [f['file'] for f in self.train_set]
        fitted = [self.descriptor_fourier(item, size) for item in train]
        return fitted

    def test(self, clf, fft_size):
        results = [self.classify_fft(
            p['file'],
            clf,
            p['class_name'],
            self.class_names,
            fft_size
        ) for p in self.test_set]
        return results

    def classify_fft(self, img, clf, real_value, classes_names, size):
        val = self.descriptor_fourier(img, size)
        res = classes_names[find_nearest(clf, val)]
        return 1 if res == real_value else 0

    @staticmethod
    def descriptor_fourier(img, size):
        img_fft = np.fft.fft2(img)
        spectrum = np.log(1 + np.abs(img_fft))
        out = []
        for i in range(0, size):
            tmp = []
            for j in range(0, size):
                tmp.append(spectrum[i][j])
            out.append(tmp)
        return list(np.concatenate(out).flat)
