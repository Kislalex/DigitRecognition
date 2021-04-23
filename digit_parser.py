#!/usr/bin/python3.9
import cv2
import time
import random
import gzip

import numpy as np
from multiprocessing import Pool
from functools import partial
from configparser import ConfigParser

from NeuralNetwork.brain import Brain
from NeuralNetwork.learning_brain import *
from digit_drawing_lib import (
    draw_digit,
    draw_digit_label,
    change_draw,
    clear_workspace,
)

config = ConfigParser()
config.read("config.ini")
HEIGHT = config["main_section"].getint("HEIGHT")
WIDTH = config["main_section"].getint("WiDTH")
DIGIT_HEIGHT = config["main_section"].getint("DIGIT_HEIGHT")
LABEL_SIZE = config["main_section"].getint("LABEL_SIZE")
digit_pixels = config["main_section"].getint("digit_pixels")
is_drawing = config["main_section"].getboolean("is_drawing")


def find_best_digit(label):
    minimum = np.exp(-1) / (np.exp(1) + 9 * np.exp(1))
    maximum = -1.0
    second_maximum = -2.0
    max_digit = 0
    for i in range(len(label)):
        if label[i] > maximum:
            second_maximum = maximum
            maximum = label[i]
            max_digit = i
        elif label[i] > second_maximum:
            second_maximum = label[i]
    maximum -= minimum
    second_maximum -= minimum
    if maximum > 2 * second_maximum:
        return max_digit, 0.99
    else:
        return max_digit, maximum / (2 * second_maximum)


def draw_two_layers(workspace, history, top, left, digit_pixels):
    bigger_digit = int(digit_pixels * 1.2)
    left -= (bigger_digit - digit_pixels) // 2
    top -= (bigger_digit - digit_pixels) // 2
    dx = 3
    size = (bigger_digit - 6 * dx) // 7
    left -= size + dx
    top -= size + dx
    small_size = (size - 4 * dx) // 5
    for i in range(8):
        for j in range(5):
            k = 5 * i + j
            draw_digit(
                workspace,
                history[1][k],
                top - dx - small_size,
                left + i * (size + dx) + j * (small_size + dx),
                small_size,
            )
        draw_digit(workspace, history[0][i], top, left + i * (size + dx), size)
    left = left + size + 2 * dx + bigger_digit
    for i in range(8):
        for j in range(5):
            k = 40 + 5 * i + j
            draw_digit(
                workspace,
                history[1][k],
                top + i * (size + dx) + j * (small_size + dx),
                left + size + dx,
                small_size,
            )
        draw_digit(workspace, history[0][i + 8], top + i * (size + dx), left, size)
    top = top + size + 2 * dx + bigger_digit
    for i in range(8):
        for j in range(5):
            k = 80 + 5 * i + j
            draw_digit(
                workspace,
                history[1][k],
                top + dx + size,
                left + size - small_size - i * (size + dx) - j * (small_size + dx),
                small_size,
            )
        draw_digit(workspace, history[0][i + 16], top, left - i * (size + dx), size)
    left = left - size - 2 * dx - bigger_digit
    for i in range(8):
        for j in range(5):
            k = 120 + 5 * i + j
            draw_digit(
                workspace,
                history[1][k],
                top - i * (size + dx) + size - small_size - j * (small_size + dx),
                left - small_size - dx,
                small_size,
            )
        draw_digit(workspace, history[0][i + 16], top - i * (size + dx), left, size)


def draw_digit_with_history(workspace, brain, digit):
    history = brain.compute_with_history(digit[0])
    outputs = history[-1][0]
    brain_label = compute_label(outputs)
    brain_guess = find_best_digit(brain_label)
    left = (WIDTH - LABEL_SIZE - DIGIT_HEIGHT) // 2
    top = (HEIGHT - DIGIT_HEIGHT) // 2
    workspace = clear_workspace(top, left, DIGIT_HEIGHT)
    draw_digit_label(
        workspace, brain_label, brain_guess, HEIGHT, WIDTH - LABEL_SIZE, LABEL_SIZE
    )
    draw_digit(workspace, digit[0], top, left, DIGIT_HEIGHT)
    draw_two_layers(workspace, history, top, left, DIGIT_HEIGHT)
    print("Correct digit : {}".format(digit[1]))
    print(outputs)
    return workspace


def exp_gradient(output_, goal_):
    total_weight = np.sum(np.exp(output_))
    e_x = np.exp(output_[goal_])
    gradient = (e_x / (total_weight * total_weight)) * np.exp(output_)
    gradient[goal_] -= e_x / total_weight
    for i in range(len(output_)):
        if i != goal_:
            if output_[i] < -0.9:
                gradient[i] *= -1
            elif output_[i] < -0.8:
                gradient[i] = 0
        else:
            if output_[i] > 0.9:
                gradient[i] *= -1
            elif output_[i] > 0.8:
                gradient[i] = 0
    return gradient


def exp_energy(output_, goal_):
    total_weight = np.sum(np.exp(output_))
    e_x = np.exp(output_[goal_])
    return -e_x / total_weight


def compute_label(output_):
    total_weight = np.sum(np.exp(output_))
    vector_weights = (1.0 / total_weight) * np.exp(output_)
    return vector_weights


def learn_digit_brain(brain, tests_, workspace, n, dn, steps, scales):
    while n < len(tests_):
        print(n)
        learning_set = tests_[n : n + dn]
        rand_index = np.random.randint(len(learning_set))
        workspace = draw_digit_with_history(workspace, brain, learning_set[rand_index])
        cv2.imshow("digits", workspace)
        cv2.waitKey(1)
        n += dn
        brain = learning_brain_by_g_d(
            brain, learning_set, steps, exp_gradient, exp_energy, scales
        )
    return brain


def modify_digit(digit, x, y):
    dxy = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if digit[x, y] < 254.0 / 500.0:
        digit[x, y] = 255.0 / 500.0
        for dx, dy in dxy:
            if x + dx > 27 or x + dx < 0 or y + dy > 27 or y + dy < 0:
                continue
            digit[x + dx, y + dy] += 85.0 / 500.0
            if 170.0 / 500.0 < digit[x + dx, y + dy] < 254.0 / 500.0:
                digit[x + dx, y + dy] = 170.0 / 500.0


def read_digits_and_labels(digits, labels):
    magic_number = int.from_bytes(digits.read(4), byteorder="big")
    size = int.from_bytes(digits.read(4), byteorder="big")
    n = int.from_bytes(digits.read(4), byteorder="big")
    m = int.from_bytes(digits.read(4), byteorder="big")
    int.from_bytes(labels.read(8), byteorder="big")
    data_digits = []
    data_labels = []
    for i in range(size):
        new_digit = np.array(
            [
                float(int.from_bytes(digits.read(1), byteorder="big")) / 500.0
                for k in range(m * n)
            ]
        ).reshape((m, n))
        x = int.from_bytes(labels.read(1), byteorder="big")
        data_digits.append(new_digit.transpose())
        data_labels.append(x)
    return list(zip(data_digits, data_labels))


def draw(brain, digit, workspace, event, x, y, flags, param):
    global is_drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        is_drawing ^= True
        change_draw(workspace, HEIGHT, 0, is_drawing)
        cv2.imshow("digits", workspace)
        cv2.waitKey(1)

    elif event == cv2.EVENT_MOUSEMOVE:
        if is_drawing:
            x -= (WIDTH - LABEL_SIZE - DIGIT_HEIGHT) // 2
            y -= (HEIGHT - DIGIT_HEIGHT) // 2
            if (x < DIGIT_HEIGHT) and (y < DIGIT_HEIGHT):
                square_side = DIGIT_HEIGHT // digit_pixels
                coord_x = x // square_side
                coord_y = y // square_side
                modify_digit(digit, coord_x, coord_y)
                draw_digit(
                    workspace,
                    digit,
                    (HEIGHT - DIGIT_HEIGHT) // 2,
                    (WIDTH - LABEL_SIZE - DIGIT_HEIGHT) // 2,
                    DIGIT_HEIGHT,
                )
                cv2.imshow("digits", workspace)

    elif event == cv2.EVENT_LBUTTONUP:
        is_drawing ^= True
        change_draw(workspace, HEIGHT, 0, is_drawing)
        history = brain.compute_with_history(digit)
        outputs = history[-1][0]
        brain_label = compute_label(outputs)
        brain_guess = find_best_digit(brain_label)
        left = (WIDTH - LABEL_SIZE - DIGIT_HEIGHT) // 2
        top = (HEIGHT - DIGIT_HEIGHT) // 2
        workspace = clear_workspace(top, left, DIGIT_HEIGHT)
        draw_digit_label(
            workspace, brain_label, brain_guess, HEIGHT, WIDTH - LABEL_SIZE, LABEL_SIZE
        )
        draw_digit(workspace, digit, top, left, DIGIT_HEIGHT)
        draw_two_layers(workspace, history, top, left, DIGIT_HEIGHT)

        cv2.imshow("digits", workspace)
        cv2.waitKey(1)
