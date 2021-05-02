#!/usr/bin/python3.9
import cv2
import time
import random
import gzip
import click
from tqdm import tqdm

import numpy as np
from multiprocessing import Pool
from functools import partial
from configparser import ConfigParser

from NeuralNetwork.brain import Brain
from digit_drawing_lib import (
    draw_digit,
    draw_digit_label,
    change_draw,
    clear_workspace,
)
from digit_parser import *


config = ConfigParser()
config.read("config.ini")
HEIGHT = config["main_section"].getint("HEIGHT")
WIDTH = config["main_section"].getint("WiDTH")
DIGIT_HEIGHT = config["main_section"].getint("DIGIT_HEIGHT")
LABEL_SIZE = config["main_section"].getint("LABEL_SIZE")
digit_pixels = config["main_section"].getint("digit_pixels")
is_training = False
is_pre_testing = False
is_training_on_bad_digits = False

digit_file = "train-images-idx3-ubyte.gz"
label_file = "train-labels-idx1-ubyte.gz"
# digit_file = "t10k-images-idx3-ubyte.gz"
# label_file = "t10k-labels-idx1-ubyte.gz"


def qqq(brain, test):
    return find_best_digit(compute_label(brain.compute(test[0])[0]))


workspace = clear_workspace(
    (HEIGHT - DIGIT_HEIGHT) // 2, (WIDTH - LABEL_SIZE - DIGIT_HEIGHT) // 2, DIGIT_HEIGHT
)
change_draw(workspace, HEIGHT, 0, False)
cv2.imshow("digits", workspace)
# rule =
#       28 x 28 -> conv
# 32 of  7 x  7 -> strited
# 160 of  3 x 3 -> linear
# 30 -> linear
# 10
digit_brain = Brain()
s = "digit_brain_dump"
try:
    f = open(s, "rb")
    digit_brain.read_from_stream(f)
    f.close()
except:
    print("Creating a new brain")
    digit_brain.add_layer(1, 32, 1, 32, (4, 4))
    digit_brain.add_layer(2, 160, 1, 5, (5, 5))
    digit_brain.add_layer(0, 1, 160, 1, (160 * 9, 30))
    digit_brain.add_layer(0, 1, 1, 1, (30, 10))
    digit_brain.mutate(0.5, 0.9)

digit = np.zeros((digit_pixels, digit_pixels))

if is_training:
    digit_stream = gzip.open(digit_file, "r")
    label_stream = gzip.open(label_file, "r")
    tests_ = read_digits_and_labels(digit_stream, label_stream)
    digit_stream.close()
    label_stream.close()
    print("Begin learning : {}".format(len(tests_)))
    n = 0
    dn = 300
    step = 2
    scales = [
        0.8,
        0.2,
        0.08,
    ]
    minimal_bad = 2810
else:
    draw_with_brain = partial(draw, digit_brain, digit, workspace)
    cv2.setMouseCallback("digits", draw_with_brain)

if is_pre_testing:
    print("Extracting test, that went wrong")
    best_digit_ = partial(qqq, digit_brain)
    guesses = []
    with Pool(16) as p:
        work = p.map_async(best_digit_, tqdm(tests_))
        p.close()
        p.join()
        guesses = work.get()
    bad_tests = []
    for i in range(len(tests_)):
        if guesses[i][0] != tests_[i][1] or guesses[i][1] < 0.6:
            bad_tests.append(tests_[i])
    print("Test accuracy : {0:.2f}%".format(100.0 - len(bad_tests) * 100.0 / len(tests_)))
    minimal_bad = len(bad_tests)

while True:
    if is_training:
        testing = tests_
        random.shuffle(testing)

        digit_brain = learn_digit_brain(
            digit_brain, testing, workspace, n, dn, step, scales
        )
        if is_training_on_bad_digits:
            print("Extracting test, that went wrong")
            best_digit_ = partial(qqq, digit_brain)
            guesses = []
            with Pool(16) as p:
                work = p.map_async(best_digit_, tqdm(tests_))
                p.close()
                p.join()
                guesses = work.get()
            bad_tests = []
            for i in range(len(tests_)):
                if guesses[i][0] != tests_[i][1] or guesses[i][1] < 0.6:
                    bad_tests.append(tests_[i])
            print(
                "Test accuracy : {0:.2f}%".format(
                    100.0 - len(bad_tests) * 100.0 / len(tests_)
                )
            )
            # testing = bad_tests
            if len(bad_tests) < minimal_bad:
                minimal_bad = len(bad_tests)
                print("Saved for {}".format(minimal_bad))
                f = open(s, "wb")
                digit_brain.write_to_stream(f)
                f.close()
    else:
        k = cv2.waitKey(1)
        if k == ord("q"):
            break
        elif k == ord("c"):
            workspace = clear_workspace(
                (HEIGHT - DIGIT_HEIGHT) // 2,
                (WIDTH - LABEL_SIZE - DIGIT_HEIGHT) // 2,
                DIGIT_HEIGHT,
            )
            for i in range(digit_pixels):
                for j in range(digit_pixels):
                    digit[i][j] = 0
            change_draw(workspace, HEIGHT, 0, False)
            cv2.imshow("digits", workspace)
cv2.destroyAllWindows()
