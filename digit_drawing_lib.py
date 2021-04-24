#!/usr/bin/python3.7
import cv2
import numpy as np
import time
import random
from configparser import ConfigParser

config = ConfigParser()
config.read("config.ini")
HEIGHT = config["main_section"].getint("HEIGHT")
WIDTH = config["main_section"].getint("WiDTH")
DIGIT_HEIGHT = config["main_section"].getint("DIGIT_HEIGHT")
LABEL_SIZE = config["main_section"].getint("LABEL_SIZE")
digit_pixels = config["main_section"].getint("digit_pixels")

blue = (255, 0, 0)
black = (0, 0, 0)
light_blue = (230, 216, 173)
orange = (0, 165, 255)
green = (0, 128, 0)
red = (0, 0, 255)
gray = (211, 211, 211)
white = (255, 255, 255)


def draw_digit(img, digit, top, left, size):
    grid_size = digit.shape[0]
    square_side = size // grid_size
    cv2.line(
        img,
        (left, top),
        (left, top + size - 1),
        black,
        1,
    )
    cv2.line(
        img,
        (left, top),
        (left + size - 1, top),
        black,
        1,
    )
    for x in range(grid_size):
        for y in range(grid_size):
            color_offset = int(digit[x, y] * 500)
            color = (
                0,
                0,
                0,
            )
            if color_offset > 0:
                color = (
                    255,
                    255 - color_offset,
                    255 - color_offset,
                )
            else:
                color = (
                    255 + color_offset,
                    255 + color_offset,
                    255,
                )
            cv2.rectangle(
                img,
                (left + x * square_side + 1, top + y * square_side + 1),
                (left + (x + 1) * square_side - 1, top + (y + 1) * square_side - 1),
                color,
                -1,
            )
    cv2.line(
        img,
        (left + size - 1, top),
        (left + size - 1, top + size - 1),
        black,
        1,
    )
    cv2.line(
        img,
        (left, top + size - 1),
        (left + size - 1, top + size - 1),
        black,
        1,
    )
    cv2.imshow("digits", img)


def draw_digit_label(img, label, guess, height, left, size):
    offset = 30
    img[0:height, left : left + size] = 221 * np.ones((height, size, 3), dtype="uint8")
    cv2.line(
        img,
        (left, 0),
        (left, height),
        orange,
        3,
    )
    cv2.line(
        img,
        (left, 0),
        (left, height),
        black,
        1,
    )
    cv2.line(
        img,
        (left + size, 0),
        (left + size, height),
        orange,
        3,
    )
    cv2.line(
        img,
        (left + size, 0),
        (left + size, height),
        black,
        1,
    )
    cv2.putText(
        img,
        "Labels",
        (left + size // 2 - 25, 15),
        16,
        0.7,
        black,
        2,
    )
    cv2.putText(
        img,
        "score",
        (left + size // 2 - 22, 30),
        16,
        0.7,
        black,
        2,
    )
    bar_size = (height - 5 * offset) // 5 - offset
    for digit in range(10):
        x = left + (size // 3) * (1 + digit % 2)
        y = 2 * offset + (bar_size + offset) * (digit // 2)
        cv2.putText(img, str(digit), (x, y), 16, 0.8, light_blue, 5)
        cv2.putText(img, str(digit), (x, y), 16, 0.8, black, 1)
        dx = size // 9
        dy = 0
        fraction = (label[digit] - 1 / 18.0) * 2.3
        if fraction < 0:
            fraction = 0.01
        dy = int(fraction * bar_size)
        cv2.rectangle(img, (x, y + 5), (x + dx, y + 5 + bar_size), red, -1)
        cv2.rectangle(img, (x, y + 5), (x + dx, y + 5 + bar_size), black, 2)
        cv2.rectangle(img, (x, y + 5), (x + dx, y + 5 + dy), green, -1)
        cv2.rectangle(img, (x, y + 5), (x + dx, y + 5 + dy), black, 2)
    cv2.putText(
        img,
        "Digit :",
        (left + size // 2 - 32, height - 3 * offset),
        16,
        0.6,
        black,
        2,
    )
    cv2.putText(
        img,
        str(guess[0]),
        (left + size // 2 - 5, height - 2 * offset),
        16,
        0.8,
        black,
        3,
    )
    cv2.putText(
        img,
        str(guess[0]),
        (left + size // 2 - 5, height - 2 * offset),
        16,
        0.8,
        blue,
        1,
    )
    cv2.putText(
        img,
        "confidence:",
        (left + size // 2 - 42, height - offset - 10),
        16,
        0.5,
        black,
        2,
    )
    cv2.putText(
        img,
        "{0:.1f}%".format(guess[1] * 100),
        (left + size // 2 - 32, height - 10),
        16,
        0.8,
        black,
        3,
    )
    cv2.putText(
        img,
        "{0:.1f}%".format(guess[1] * 100),
        (left + size // 2 - 32, height - 10),
        16,
        0.8,
        red,
        1,
    )


def change_draw(img, top, left, is_drawing):
    if is_drawing:
        cv2.putText(img, "OFF", (140, 15), 16, 0.5, gray, 2)
        cv2.putText(img, "ON", (140, 15), 16, 0.5, green, 2)
    else:
        cv2.putText(img, "ON", (140, 15), 16, 0.5, gray, 2)
        cv2.putText(img, "OFF", (140, 15), 16, 0.5, red, 2)


def clear_workspace(digit_top, digit_left, digit_size):
    img = 221 * np.ones((HEIGHT, WIDTH, 3), dtype="uint8")
    img[
        digit_top : digit_top + digit_size, digit_left : digit_left + digit_size
    ] = 255 * np.ones((digit_size, digit_size, 3), dtype="uint8")
    square_side = digit_size // digit_pixels
    for i in range(digit_pixels + 1):
        dx = int(i * square_side)
        cv2.line(
            img,
            (digit_left, digit_top + dx),
            (digit_left + digit_size, digit_top + dx),
            light_blue,
            1,
        )
        cv2.line(
            img,
            (digit_left + dx, digit_top),
            (digit_left + dx, digit_top + digit_size),
            light_blue,
            1,
        )
    cv2.putText(img, "mouse_draw:", (10, 15), 16, 0.5, black, 1)
    cv2.putText(
        img, "to clear the image press 'c'", (10, HEIGHT - 10), 16, 0.5, black, 1
    )
    return img
