# -*- coding: utf-8 -*-
"""
@Version: 0.1
@Author: Charles
@Time: 2022/10/10 18:32
@File: io_utils.py
@Desc: 
"""
import yaml


def write_yaml(data, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, Dumper=yaml.SafeDumper)
