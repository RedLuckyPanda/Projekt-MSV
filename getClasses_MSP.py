#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import arff
import numpy as np

classes = {}

def get_MSP_dataset():

    with open("Evaluation.txt") as file:
        for line in file:
            if "MSP" in line:
                name = line.split()[0]
                name = name[:-1]
                emotion = line.split()[1]
                emotion = emotion[0]
                classes[name] = emotion

    output = []
    targetdata = []
    dataset = arff.load(open("mspAudio_all_gemaps.arff"))
    remove_classes = ["O", "X", "H", "N"]
    for element in dataset['data']:
        element[-1] = classes[element[0]]


        if (element[-1] not in remove_classes):
            if (element[0].__contains__("-P-") == False):
                output.append(element[1:-1])
                targetdata.append(element[-1])

    array_data = np.array(output)
    array_target = np.array(targetdata)

    return array_data, array_target

#get_MSP_dataset()

