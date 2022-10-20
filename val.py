# -*- coding: utf-8 -*-
"""
@Version: 0.1
@Author: Charles
@Time: 2022/10/19 20:51
@File: val.py
@Desc: 
"""
import os
import pickle

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import resnet

label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


@torch.no_grad()
def val(model_path, save_dir):
    os.makedirs(save_dir,exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    test_transform = transforms.Compose([
        # transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # 模型
    model_name = model_path.split('/')[-3].split('-')[0]
    model = getattr(resnet, model_name)(num_classes=10)
    model.load_state_dict(torch.load(model_path, map_location='cpu')['state_dict'])
    model.to(device)
    model.eval()

    data = []
    targets = []
    file_path = './data/cifar-10-batches-py/test_batch'
    with open(file_path, 'rb') as f:
        entry = pickle.load(f, encoding='latin1')
        data.append(entry['data'])
        if 'labels' in entry:
            targets.extend(entry['labels'])
        else:
            targets.extend(entry['fine_labels'])

    data = np.vstack(data).reshape(-1, 3, 32, 32)
    data = data.transpose((0, 2, 3, 1))  # convert to HWC

    acc = 0
    for index, (image, target) in enumerate(zip(data, targets)):
        image = Image.fromarray(image)
        src_image = image.copy()
        image = test_transform(image)
        image = torch.unsqueeze(image, 0)
        image = image.to(device)
        logits = model(image)
        probs = torch.softmax(logits, dim=1)
        _, preds = torch.max(probs, 1)
        cls = int(preds[0])
        pred_name = label_names[cls]
        target_name = label_names[target]
        if pred_name == target_name:
            acc += 1
        else:
            src_image.save(os.path.join(save_dir, f'{index}-{target_name}-{pred_name}.jpg'))
    print(acc, len(data), acc / len(data))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    model_path = './output/resnet18-2022-10-19-19-20-51/model/resnet18-best.pth'
    save_dir = './data/err_case'
    val(model_path, save_dir)