from skimage import io

import numpy as np
import os
import cv2


def find_nearest(array, value):
    dist = []
    array = np.asarray(array)
    value = np.asarray(value)
    for i in range(len(array)):
        dist.append(np.linalg.norm(array[i] - value))
    dist = np.asarray(dist)
    idx = dist.argmin()
    return idx


class UnlFourierDescriptor:
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
        fitted = [self.unl_fourier_desc(item, size) for item in train]
        return fitted

    def unl_fourier_desc(self, img, size):
        # cartesian coordinates of image to polar coordinates
        thresh = cv2.Canny(img, 30, 200)
        value = np.sqrt(((thresh.shape[0] / 2.0) ** 2.0) + ((thresh.shape[1] / 2.0) ** 2.0))
        polar_image = cv2.linearPolar(thresh, (thresh.shape[0] / 2, thresh.shape[1] / 2), value,
                                      cv2.WARP_FILL_OUTLIERS)
        polar_image = polar_image.astype(np.uint8)
        # cv2.imshow("Image", thresh)
        # cv2.imshow("Polar Image", polar_image)
        # cv2.waitKey(0)

        # fourier
        img_fft = np.fft.fft2(polar_image)
        spectrum = np.log(1 + np.abs(img_fft))
        # plt.imshow(spectrum, "gray")
        # plt.title("Spectrum")
        # plt.show()
        out = []
        for i in range(0, size):
            tmp = []
            for j in range(0, size):
                tmp.append(spectrum[i][j])
            out.append(tmp)
        return list(np.concatenate(out).flat)

    def test(self, clf, s):
        results = []
        for img in self.test_set:
            results.append(self.classify(img, clf, s))
        return results

    def classify(self, img, train, s):
        tmp = self.unl_fourier_desc(img['file'], s)
        res = self.enc_idx(find_nearest(train, tmp))
        out = []
        # print("Deskryptor - UNL-Fourier: ", res)
        out.append(1) if res == img['class_name'] else out.append(0)
        return out

    def enc_idx(self, i):
        return self.class_names[i]

    def score(self, scores):
        res = 0
        for i in range(len(scores)):
            res += scores[i][0]
        res = res / len(scores) * 100
        return res
