#!/usr/bin/env python3
import os
import pickle

# from matplotlib import pyplot as plt

from tqdm import tqdm
import torch

from torchvision import datasets
# from coco_utils import get_coco

import transforms as T

data_root_dir = os.environ["DATA_PATH"]


def get_summary_transforms():
    transforms = []
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]))

    return T.Compose(transforms)


sbd_train = datasets.SBDataset(data_root_dir+"/sbd/", image_set='train', mode='segmentation', download=False, transforms=get_summary_transforms())
sbd_val = datasets.SBDataset(data_root_dir+"/sbd/", image_set='val', mode='segmentation', download=False, transforms=get_summary_transforms())

cityscapes_train = datasets.Cityscapes(data_root_dir+"/cityscapes/", split='train', mode='fine', target_type='semantic', transforms=get_summary_transforms())
cityscapes_val = datasets.Cityscapes(data_root_dir+"/cityscapes/", split='val', mode='fine', target_type='semantic', transforms=get_summary_transforms())

# coco_train = get_coco(data_root_dir+'/coco/', image_set='train', transforms=get_summary_transforms())
# coco_val = get_coco(data_root_dir+'/coco/', image_set='val', transforms=get_summary_transforms())

voc_train = datasets.VOCSegmentation(data_root_dir+"/PascalVOC2012/", year='2012', image_set='train', download=False, transforms=get_summary_transforms())
voc_val = datasets.VOCSegmentation(data_root_dir+"/PascalVOC2012/", year='2012', image_set='val', download=False, transforms=get_summary_transforms())


datasets = {
        'sbd': {'train': sbd_train, 'val': sbd_val},
        # 'coco' : {'train' : coco_train, 'val': coco_val},
        'voc': {'train': voc_train, 'val': voc_val},
        'cityscapes': {'train': cityscapes_train, 'val': cityscapes_val},
        }

results = {}
for name in datasets:
    if name == 'cityscapes':
        classes = [
                'unlabeled', 'ego vehicle', 'rectification border', 'out of roi', 'static', 'dynamic', 'ground', 'road', 'sidewalk',
                'parking', 'rail track', 'building', 'wall', 'fence', 'guard rail', 'bridge', 'tunnel', 'pole', 'polegroup',
                'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'caravan',
                'trailer', 'train', 'motorcycle', 'bicycle',
                ]
        num_classes = len(classes)
    else:
        num_classes = 21

        classes = [
                '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
                ]
    print(name)
    results[name] = {}
    for image_set in ['train', 'val']:
        dataset = datasets[name][image_set]
        print('\t', image_set)
        print('samples: ', len(dataset))
        labelcounts = torch.zeros(num_classes, dtype=torch.int64)
        # resprint = False
        # i = 0
        for img, target in tqdm(dataset, disable=False):
            # if not resprint:
            #     print(target.size())
            #     resprint=True
            # if i > 10:
            #     break
            # i += 1

            sample_labelcounts = torch.bincount(torch.flatten(target), minlength=num_classes)
            labelcounts += sample_labelcounts[:num_classes]  # NOTE for 255 means unlabeled
        print(labelcounts.double()/labelcounts.sum())
        print('pixels: ', labelcounts.sum().item())
        results[name][image_set] = labelcounts.double()/labelcounts.sum()

        # plt.clf()
        # plt.bar(np.arange(len(classes)), labelcounts.double()/labelcounts.sum(), align='center')
        # plt.xticks(np.arange(len(classes)), classes, rotation=90)
        # plt.ylabel('relative occurence')
        # plt.yscale('log')
        # plt.title(name + ' ' + image_set + ' class frequencies')
        # plt.tight_layout()

        # plt.savefig('plots/' + name + '_' + image_set + '_clsfreq.pdf')

print(results.keys())
print(results)
with open('class_freq.pkl', 'wb') as f:
    pickle.dump(results, f)
