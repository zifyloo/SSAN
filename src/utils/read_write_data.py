# -*- coding: utf-8 -*-
"""
the tool to read or write the data. Have a good luck !

@author: zifyloo
"""

import os
import json
import pickle


def makedir(root):
    if not os.path.exists(root):
        os.makedirs(root)


def write_json(data, root):
    with open(root, 'w') as f:
        json.dump(data, f)


def read_json(root):
    with open(root, 'r') as f:
        data = json.load(f)

    return data


def read_dict(root):
    with open(root, 'rb') as f:
        data = pickle.load(f)

    return data


def save_dict(data, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def write_txt(data, name):
    with open(name, 'a') as f:
        f.write(data)
        f.write('\n')




