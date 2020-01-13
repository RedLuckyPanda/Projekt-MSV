#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 16:13:43 2019

@author: zywietbe
"""

import arff
import numpy as np

def arff_dataToArray(arffdata = ""):
    targetdata = []
    if arffdata == "":
        arffdata = "audio_all_gemaps.arff"
    dataset = arff.load(open(arffdata))
    output = []
    # Daten insgesamt = 6277
    # größte Klasse = 1849
    # 0 - anger             +
    # 1 - disgust
    # 2 - excited           +
    # 3 - fear
    # 4 - frustration (für besseren Vergleich mit msp-improv ausgeschlossen)
    # 5 - happiness        (+)
    # 6 - neutral state     +
    # 7 - other
    # 8 - sad               +
    # 9 - surprise
    # 10 - xxx - no ground truth across evaluators
    for element in dataset["data"]:
        keep_classes = [0,2,5]
        if (element[-1] in keep_classes):
            if (element[-1] == 5):
                element[-1] = 2
            output.append((element[1:-1]))
            targetdata.append(element[-1])

    array_data = np.array(output)
    array_target = np.array(targetdata)
    return array_data, array_target
