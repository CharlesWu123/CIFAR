# -*- coding: utf-8 -*-
"""
@Version: 0.1
@Author: Charles
@Time: 2022/10/19 13:37
@File: data_process.py
@Desc: 
"""
import cv2
import numpy as np

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def show_img(file_path):
    '''
    dict_keys([b'batch_label', b'labels', b'data', b'filenames'])
    Args:
        file_path:
    Returns:
    '''
    d = unpickle(file_path)
    data = d[b'data']
    labels = d[b'labels']
    filenames = d[b'filenames']
    batch_label = d[b'batch_label']
    for index, (label, filename) in enumerate(zip(labels, filenames)):
        print('label:{}, filename: {}'.format(classes[label], filename.decode(encoding='utf-8')))
        img = data[index].reshape(3, 32, 32)
        img = img.transpose(1, 2, 0)            # cwh->whc
        img = img[:, :, ::-1]                   # rgb -> bgr
        cv2.imwrite('./examples/' + filename.decode(encoding='utf-8'), img)
        if index > 10:
            break


if __name__ == '__main__':
    # d = unpickle('./cifar-10-batches-py/data_batch_1')
    # print(d.keys())

    show_img('./cifar-10-batches-py/data_batch_1')