import torch
import torch.utils.data as data
from PIL import Image
from spatial_transforms import *
import os
import math
import functools
import json
import copy
from numpy.random import randint
import numpy as np
import random
import matplotlib.pyplot as plt
from utils import load_value_file
import pdb
import torch.nn.functional as F
from matplotlib import cm



from augmentations import *
from transforms import *
from utils_ import *
from datasets_ import *
from models_ import *

import os


from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2


class Yolo_processing:
    def __init__(self):
        self.model_def="/Users/siddhantgupta/Desktop/IPN_hand/datasets/config/yolov3_testing.cfg"
        self.weights_path="/Users/siddhantgupta/Desktop/IPN_hand/datasets/yolov3_training_final 3.weights"
        self.class_path="/Users/siddhantgupta/Desktop/IPN_hand/datasets/yolov3_training_final 3.weights"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_cpu = 0
        self.img_size = 416
        self.model = Darknet(self.model_def, img_size=self.img_size).to(self.device)
        self.conf_thres=0.1
        self.nms_thres=0.1
        if self.weights_path.endswith(".weights"):
            # Load darknet weights
            self.model.load_darknet_weights(self.weights_path)
        else:
            # Load checkpoint weights
            self.model.load_state_dict(torch.load(self.weights_path,map_location=torch.device('cpu')))
        self.model.eval()  # Set in evaluation
        self.model.share_memory()

        print('Yolo Model Loaded...')

    def yolo_eval(self,input_lis):

        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        width, height = input_lis[0].size
        cmap = plt.get_cmap("tab20b")
        colors = [cmap(i) for i in np.linspace(0, 1, 20)]
        resulting_clip = []

        dataloader = DataLoader(
            ImageFolder(input_lis, transform= transforms.Compose([DEFAULT_TRANSFORMS, Resize(self.img_size)])),
            batch_size=1,
            shuffle=False,
            num_workers=self.n_cpu,
        )
        for batch_i, (img,original_img) in enumerate(dataloader):
            original_img = np.squeeze(original_img, axis=0)
            input_imgs = Variable(img.type(Tensor))
            original_img = original_img.numpy()



            with torch.no_grad():
                detections = self.model(input_imgs)
                detections = non_max_suppression(detections, self.conf_thres, self.nms_thres)

            detections = detections[0]

            if detections is not None:
                detections = rescale_boxes(detections, self.img_size, original_img.shape[:2])
                unique_labels = detections[:, -1].cpu().unique()
                n_cls_preds = len(unique_labels)
                bbox_colors = random.sample(colors, n_cls_preds)

                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    
                    x1 = int(x1)
                    y1 = int(y1)
                    x2 = int(x2)
                    y2 = int(y2)

                    mask = np.zeros_like(original_img)
                    mask = cv2.rectangle(mask, (x1, y1), (x2,y2),  (255,255,255), -1)

                    result = cv2.bitwise_and(original_img, mask)
                    result = Image.fromarray(result)
                    result = result.resize((width, height))
            else:
                result = np.zeros((img.shape[0],img.shape[1],3),np.uint8)
                result = Image.fromarray(result)
                result = result.resize((width, height))

            resulting_clip.append(result)
        return resulting_clip

def pil_loader(path, modality):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        #print(path)
        with Image.open(f) as img:
            if modality in ['RGB', 'flo']:
                return img.convert('RGB')
            elif modality in ['Depth', 'seg']:
                return img.convert('L') # 8-bit pixels, black and white check from https://pillow.readthedocs.io/en/3.0.x/handbook/concepts.html


def accimage_loader(path, modality):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    return pil_loader
    # from torchvision import get_image_backend
    # if get_image_backend() == 'accimage':
    #     return accimage_loader
    # else:
    #     return pil_loader


def video_loader(video_dir_path, frame_indices, modality, sample_duration, image_loader):
    
    video = []
    if modality in ['RGB', 'flo', 'seg']:
        for i in frame_indices:
            image_path = os.path.join(video_dir_path, '{:s}_{:06d}.jpg'.format(video_dir_path.split('/')[-1],i))
            if os.path.exists(image_path):
                
                video.append(image_loader(image_path, modality))
            else:
                print(image_path, "------- Does not exist")
                return video
    elif modality in ['RGB-flo', 'RGB-seg']:
        for i in frame_indices: # index 35 is used to change img to flow
        # seg1CM42_21_R_#156_000076
            image_path = os.path.join(video_dir_path, '{:s}_{:06d}.jpg'.format(video_dir_path.split('/')[-1],i))

            if modality.split('-')[1] == 'flo':
                sensor = 'flow'
            elif modality.split('-')[1] == 'seg':
                sensor = 'segment'
            image_path_depth = os.path.join(video_dir_path.replace('frames',sensor), '{:s}_{:06d}.jpg'.format(video_dir_path.split('/')[-1],i))
            
            image = image_loader(image_path, 'RGB')
            image_depth = image_loader(image_path_depth, 'Depth')

            if os.path.exists(image_path):
                video.append(image)
                video.append(image_depth)
            else:
                print(image_path, "------- Does not exist")
                return video
    
    return video

def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def load_annotation_data(data_file_path):
    # print(os.getcwd())
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            label = value['annotations']['label']
            #video_names.append('{}/{}'.format(label, key))
            video_names.append(key.split('^')[0])
            annotations.append(value['annotations'])

    return video_names, annotations


def make_dataset(root_path, annotation_path, subset, n_samples_for_each_video, sample_duration):
    data = load_annotation_data(annotation_path)
    video_names, annotations = get_video_names_and_annotations(data, subset)
    class_to_idx = get_class_labels(data)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    dataset = []
    print("[INFO]: IPN Dataset - " + subset + " is loading...")
    print("  path: " + video_names[0])
    for i in range(len(video_names)):
        if i % 1000 == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_names)))

        video_path = os.path.join(root_path, video_names[i])
        if not os.path.exists(video_path):
            continue

        begin_t = int(annotations[i]['start_frame'])
        end_t = int(annotations[i]['end_frame'])
        n_frames = end_t - begin_t + 1
        sample = {
            'video': video_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            #'video_id': video_names[i].split('/')[1]
            'video_id': i
        }
        if len(annotations) != 0:
            sample['label'] = class_to_idx[annotations[i]['label']]
        else:
            sample['label'] = -1

        if n_samples_for_each_video == 1:
            sample['frame_indices'] = list(range(begin_t, end_t + 1))
            dataset.append(sample)
        else:
            if n_samples_for_each_video > 1:
                step = max(1,
                           math.ceil((n_frames - 1 - sample_duration) /
                                     (n_samples_for_each_video - 1)))
            else:
                step = sample_duration
            for j in range(1, n_frames, step):
                sample_j = copy.deepcopy(sample)
                sample_j['frame_indices'] = list(
                    range(j, min(n_frames + 1, j + sample_duration)))
                dataset.append(sample_j)
    # print(dataset,idx_to_class)
    return dataset, idx_to_class

class IPN(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 modality='RGB',
                 get_loader=get_default_video_loader):
        self.data, self.class_names = make_dataset(
            root_path, annotation_path, subset, n_samples_for_each_video,
            sample_duration)
        self.yolo=Yolo_processing()
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.modality = modality
        self.sample_duration = sample_duration
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        path = self.data[index]['video']
        # print(patxh)
        frame_indices = self.data[index]['frame_indices']


        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(path, frame_indices, self.modality, self.sample_duration)
        count = 1

        for img in clip:
          count = count +1
          img.save('images/before'+path.split('/')[-1]+str(count)+'.png')
        clip = self.yolo.yolo_eval(clip)
        count = 1
        for img in clip:
          print(np.array(img).shape)
          count = count +1
          img.save('images/after'+path.split('/')[-1]+str(count)+'.png')
        
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]


        for img in clip:
            count = count +1
            print(img.shape)
            im = img.permute(1,2,0).numpy()
            print(im.shape)
            # im.save('images/transform'+path.split('/')[-1]+str(count)+'.png')
            im = Image.fromarray(np.uint8(im*255))
            im.save('images/transform'+path.split('/')[-1]+str(count)+'.png')
        
        im_dim = clip[0].size()[-2:]
        clip = torch.cat(clip, 0).view((self.sample_duration, -1) + im_dim).permute(1, 0, 2, 3)
        
     
        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)
        print(target,self.data[index])


        return clip, target

    def __len__(self):
        return len(self.data)


