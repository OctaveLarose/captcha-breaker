import os
import shutil
import cv2
import imutils
import numpy as np
from typing import Tuple

DATASET_DIR = './dataset'
CHARS_DIR = './chars'
char_counts = {}


def get_mod_imgs(captcha_image_file: str):
    image = cv2.imread(captcha_image_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # TODO: is it still necessary? Adds some extra padding around the image
    # gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)

    # threshold the image (convert it to pure black and white)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # eroding/dilating it to remove noise
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    mod_thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, element, iterations=3)

    # Gives +10 finds
    kernel = np.ones((2, 2), np.uint8)
    mod_thresh = cv2.morphologyEx(mod_thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    return mod_thresh, gray


def get_contours(mod_thresh):
    contours = cv2.findContours(mod_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Hack for compatibility with different OpenCV versions
    return contours[1] if imutils.is_cv3() else contours[0]


def get_letter_image_regions(contours):
    def split_contour_in_n(ct: Tuple, n: int) -> [Tuple]:
        split_contour = []
        for i in range(n):
            split_contour.append((ct[0] + i * ct[2] // n, ct[1], ct[2] // n, ct[3]))

        return split_contour

    letter_image_regions = []

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        # Removing contours that are too small
        if w * h < 200:
            continue

        letter_image_regions.append((x, y, w, h))


    # Splitting the biggest one into two if there are three
    if len(letter_image_regions) == 3:
        letter_image_regions = sorted(letter_image_regions, key=lambda a: a[2])

        # Getting the one with the highest width from the back of the sorted list...
        # ...and splitting it in two
        letter_image_regions += split_contour_in_n(letter_image_regions.pop(), 2)


    # If there are two, choose between splitting both into two or splitting the biggest one into three.
    # Splitting the biggest one into two if there are three
    if len(letter_image_regions) == 2:
        letter_image_regions = sorted(letter_image_regions, key=lambda a: a[2])

        w1, w2 = letter_image_regions[0][2], letter_image_regions[1][2]

        # If their size is similar: split both into two
        TWO_SPLIT_THRES = 0.8
        if 1 - TWO_SPLIT_THRES < w1 / w2 < 1 + TWO_SPLIT_THRES:
            ct1_split = split_contour_in_n(letter_image_regions.pop(), 2)
            ct2_split = split_contour_in_n(letter_image_regions.pop(), 2)
            letter_image_regions += (ct1_split + ct2_split)
        else:
            # Else, splitting the biggest one into three.
            letter_image_regions += split_contour_in_n(letter_image_regions.pop(), 3)


    # A single block needs to be split into 4.
    if len(letter_image_regions) == 1:
        letter_image_regions += split_contour_in_n(letter_image_regions.pop(), 4)


    # Sort the detected letter images based on the x coordinate to make sure
    # we are processing them from left-to-right so we match the right image
    # with the right letter
    letter_image_regions = sorted(letter_image_regions, key=lambda a: a[0])
    return letter_image_regions


def save_chars_to_imgs(letter_image_regions: [Tuple[int]], captcha_correct_text: str, main_img: [[]]) -> None:
    # Saving individual characters to images
    for char_box, char_text in zip(letter_image_regions, captcha_correct_text):
        # Grab the coordinates of the letter in the image
        x, y, w, h = char_box

        # Extract the letter from the original image with a 2-pixel margin around the edge
        letter_image = main_img[y - 2 if y - 2 > 0 else 0:y + h + 2, x - 2 if x - 2 > 0 else 0:x + w + 2]

        if not letter_image.any():
            break

        # Get the folder to save the image in
        save_path = os.path.join(CHARS_DIR, char_text)

        # if the output directory does not exist, create it
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # write the letter image to a file
        count = char_counts.get(char_text, 1)
        p = os.path.join(save_path, f"{str(count)}_{captcha_correct_text}.png")
        cv2.imwrite(p, letter_image)

        # increment the count for the current key
        char_counts[char_text] = count + 1


def main():
    if os.path.exists(CHARS_DIR):
        shutil.rmtree(CHARS_DIR)
    os.makedirs(CHARS_DIR)

    nbr_valid = 0
    for idx, filename in enumerate(os.listdir(DATASET_DIR)):
        captcha_image_file = os.path.join(DATASET_DIR, filename)

        print('Processing image {} ({}/{})'.format(captcha_image_file, idx, len(os.listdir(DATASET_DIR))))

        mod_thresh, gray = get_mod_imgs(captcha_image_file)
        contours = get_contours(mod_thresh)
        letter_image_regions = get_letter_image_regions(contours)

        # for char_box in letter_image_regions:
        #     x, y, w, h = char_box
        #     cv2.rectangle(mod_thresh, (x - 2, y - 2), (x + w + 4, y + h + 4), (255, 255, 255), 1)

        if len(letter_image_regions) != 4:
            # print(filename)
            # cv2.imwrite("out.png", mod_thresh)
            # cv2.imshow("Output", mod_thresh)
            # key = cv2.waitKey() & 0xFF
            # if key == ord("q"):
            #     exit()
            continue
        # print(filename)
        # cv2.imwrite("out.png", mod_thresh)
        # cv2.imshow("Output", mod_thresh)
        # key = cv2.waitKey() & 0xFF
        # if key == ord("q"):
        #     exit()

        save_chars_to_imgs(letter_image_regions, os.path.splitext(filename)[0], mod_thresh)
        nbr_valid += 1

    print('{} images had four sections precisely.'.format(nbr_valid))


if __name__ == '__main__':
    main()
