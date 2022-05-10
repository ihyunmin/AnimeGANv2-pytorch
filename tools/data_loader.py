import os
import tensorflow as tf
import cv2,random
import numpy as np
from torch.utils.data import Dataset


class ImageDataset(Dataset):

    def __init__(self, image_dir):
        self.paths = self.get_image_paths_train(image_dir)
        self.num_images = len(self.paths)

    def get_image_paths_train(self, image_dir):
        paths = []
        for path in os.listdir(image_dir):
            # Check extensions of filename
            if path.split('.')[-1] not in ['jpg', 'jpeg', 'png', 'gif']:
                continue
            # Construct complete path to anime image
            path_full = os.path.join(image_dir, path)

            # Validate if colorized image exists
            if not os.path.isfile(path_full):
                continue

            paths.append(path_full)
        
        return paths

    def read_image(self, img_path1):

        if 'style' in img_path1 or 'smooth' in img_path1:
            # color image1
            image1 = cv2.imread(img_path1).astype(np.float32)
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
            image1 = np.transpose(image1, (2,0,1))
            

            # gray image2
            image2 = cv2.imread(img_path1,cv2.IMREAD_GRAYSCALE).astype(np.float32)
            image2 = np.asarray([image2,image2,image2])
            # image2= np.transpose(image2,(1,2,0))
        else:
            # color image1
            image1 = cv2.imread(img_path1).astype(np.float32)
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
            image1 = np.transpose(image1, (2,0,1))

            image2 = np.zeros(image1.shape).astype(np.float32)

        return image1, image2

    def load_image(self, img1):
        image1, image2 = self.read_image(img1)
        processing_image1 = image1/ 127.5 - 1.0
        processing_image2 = image2/ 127.5 - 1.0
        return (processing_image1,processing_image2)
    
    def __len__(self):
        return self.num_images

    def __getitem__(self, index):
        image1, image2 = self.read_image(self.paths[index])
        processing_image1 = image1/ 127.5 - 1.0
        processing_image2 = image2/ 127.5 - 1.0
        return (processing_image1,processing_image2)
    