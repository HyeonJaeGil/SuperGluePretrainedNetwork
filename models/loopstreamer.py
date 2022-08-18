import os
from os.path import join, exists
import sys
import cv2
from matplotlib import pyplot as plt
import numpy as np
import copy
import csv

class ImagePreprocessor:
    def __init__(self, K=None, D=None, h=None, w=None, sigma=1.0):
        self.K = K
        self.D = D    
        self.h = h
        self.w = w
        self.sigma = sigma

    def undistort_image(self, cv_img):
        if self.K is None or self.D is None:
            return cv_img
        undist_img = cv2.undistort(cv_img, self.K, self.D, None)
        return undist_img

    def crop_image(self, img, deepcopy=False):
        if self.h is None or self.w is None:
            return img
        h_orig, w_orig = img.shape
        assert h_orig >= self.h and w_orig >= self.w

        if deepcopy:
            img_clone = copy.deepcopy(img)
            crop_img = img_clone[0:self.h, 0:self.w].astype('uint16')
        else:
            crop_img = img[0:self.h, 0:self.w].astype('uint16')
            
        return crop_img

    def normalize_img_with_meanstd(self, img):
        mean, std = np.mean(img), np.std(img)
        min, max = mean - self.sigma* std, mean + self.sigma* std
        
        img = (img - min) / (max - min)
        np.clip(img, 0, 1, out=img)
        img = cv2.normalize(img, None, 0, 255, 
                            cv2.NORM_MINMAX).astype('uint8')
        return img

    def preprocess_image(self, image, undistort=False, crop=False, normalize=True):
        if undistort:
            image = self.undistort_image(image)
        if crop:
            image = self.crop_image(image, deepcopy=True)
        if normalize:
            image = self.normalize_img_with_meanstd(image)
        return image
        


class LoopPairLoader:
    def __init__(self, loop_pair, root_path, img_folder, image_preprocessor, looptype=1):
        self.load_pairs_from_csv(loop_pair)
        self.root_path = root_path
        self.img_folder = img_folder
        self.img_preprocessor = image_preprocessor
        
        self.looptype = looptype
        if self.looptype == 0:
            self.loop_pairs = self.totalloop_pairs
        elif self.looptype == 1:
            self.loop_pairs = self.interloop_pairs
        elif self.looptype == 2:
            self.loop_pairs = self.intraloop_pairs

        self.count = 0
        self.nonexist_count = 0

    def size(self):
        return self.loop_pairs.shape[0]

    def where(self):
        return (self.count, self.count / self.loop_pairs.shape[0])

    def to2digit(self, input):
        number = str(input).split('.')[0]
        assert '.' not in number
        return ('0' * (2-len(number)) + number)

    def to6digit(self, input):
        number = str(input).split('.')[0]
        assert '.' not in number
        return ('0' * (6-len(number)) + number)

    def get_img_list_from_path(self, img_path):
        img_list = sorted(os.listdir(img_path))
        print(len(img_list))
        return [join(img_path,img) for img in img_list]

    def load_pairs_from_csv(self, loop_pair_csv):
        # read loop pair
        with open(loop_pair_csv, 'r') as file:
            csvreader = csv.reader(file)
            total = []
            interloop = []
            intraloop = []
            for row in csvreader:
                total.append(
                    [self.to6digit(row[0]), self.to6digit(row[1]), 
                    self.to2digit(row[2]), self.to2digit(row[3]), row[4]])
                if row[2] == row[3]:
                    intraloop.append(
                        [self.to6digit(row[0]), self.to6digit(row[1]), 
                        self.to2digit(row[2]), self.to2digit(row[3]), row[4]])
                else:
                    interloop.append(
                        [self.to6digit(row[0]), self.to6digit(row[1]), 
                        self.to2digit(row[2]), self.to2digit(row[3]), row[4]])

        self.totalloop_pairs = np.asarray(total, dtype='str')
        self.intraloop_pairs = np.asarray(intraloop, dtype='str')
        self.interloop_pairs = np.asarray(interloop, dtype='str')


    def next_pair(self):
        if self.count == self.loop_pairs.shape[0]-1:
            return (None, None, None, None, False)
        # print("count:", self.count)
        
        pair = self.loop_pairs[self.count]
        img1_path = join(self.root_path, pair[2], self.img_folder, pair[0]+'.png')
        img2_path = join(self.root_path, pair[3], self.img_folder, pair[1]+'.png')
        
        if not exists(img1_path) or not exists(img2_path):
            self.nonexist_count += 1
            if self.nonexist_count > 10:
                return (None, None, None, None, False)
            else:
                return (None, None, None, None, True)
        
        img1 = cv2.imread(img1_path, cv2.IMREAD_UNCHANGED).astype('float32')
        img2 = cv2.imread(img2_path, cv2.IMREAD_UNCHANGED).astype('float32')
        img1 = self.img_preprocessor.normalize_img_with_meanstd(img1)
        img2 = self.img_preprocessor.normalize_img_with_meanstd(img2)

        self.count += 1
        return (img1, img2, pair[2] + '_' + pair[0], pair[3] + '_' + pair[1], True)

    def plot_loop_pairs(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        for pair in self.loop_pairs:
            # if pair[2] == pair[3] and float(pair[4])> 0.5: # for intra-loop pairs
            if pair[2] != pair[3] and float(pair[4])> 0.5: # for inter-loop pairs                
                img1_path = join(self.root_path, pair[2], self.img_folder, pair[0]+'.png')
                img2_path = join(self.root_path, pair[3], self.img_folder, pair[1]+'.png')
                
                # TODO: if file not exist 10 times, break.
                if not exists(img1_path) or not exists(img2_path):
                    continue

                img1 = cv2.imread(img1_path, cv2.IMREAD_UNCHANGED).astype('float32')
                img2 = cv2.imread(img2_path, cv2.IMREAD_UNCHANGED).astype('float32')
                img1 = self.img_preprocessor.normalize_img_with_meanstd(img1)
                img2 = self.img_preprocessor.normalize_img_with_meanstd(img2)

                ax1.clear()
                ax2.clear()
                ax1.title.set_text(pair[2]+'(' + pair[0] + ')')
                ax2.title.set_text(pair[3]+'(' + pair[1] + ')')
                ax1.imshow(img1, cmap='gray')
                ax2.imshow(img2, cmap='gray')
                plt.draw()
                plt.pause(0.1)
