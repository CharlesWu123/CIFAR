# -*- coding: utf-8 -*-
'''
@Version : 0.1
@Author : Charles
@Time : 2022/10/7 15:35 
@File : test.py 
@Desc : 
'''
import os
import time

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
# from torchvision.models import resnet
import resnet


label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


@torch.no_grad()
def predict(img_path, model_path):
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

    image = Image.open(img_path)
    image = test_transform(image)
    image = torch.unsqueeze(image, 0)
    image = image.to(device)

    start_time = time.time()
    logits = model(image)
    end_time = time.time()
    probs = torch.softmax(logits, dim=1)
    _, preds = torch.max(probs, 1)
    cls = int(preds[0])
    pred_name = label_names[cls]
    return pred_name, round(float(probs[0][cls]), 2), round(end_time-start_time, 2)


def plot_one_box_PIL(img, color=None, label=None, line_thickness=None):
    w, h = img.size
    draw = ImageDraw.Draw(img)
    line_thickness = line_thickness or max(int(min(img.size) / 200), 2)
    fontsize = max(round(max(img.size) / 20), 12)
    font = ImageFont.truetype("./utils/simfang.ttf", fontsize)
    txt_width, txt_height = font.getsize(label)
    if 4 - txt_height < 0:
        draw.rectangle([0, 0, txt_width, txt_height - 4], fill=tuple(color))
        draw.text((0, 0 - 3), label, fill=(255, 255, 255), font=font)
    else:
        draw.rectangle([0, 4-txt_height, txt_width, 0], fill=tuple(color))
        draw.text((0, 1 - txt_height), label, fill=(255, 255, 255), font=font)
    return img


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    img_path = './data/examples/camion_s_000148.png'
    model_path = './output/resnet18-2022-10-19-19-20-51/model/resnet18-best.pth'
    pred_name, prob, t = predict(img_path, model_path)
    print('预测结果：{}, 预测概率：{:.2f}, 预测时间: {:.2f}'.format(pred_name, prob, t))
    # 画出结果
    draw_img = plot_one_box_PIL(Image.open(img_path), color=(0, 0, 255), label=f'{pred_name} {prob}')
    draw_img.save('./data/res.jpg')
