import os
import shutil
import cv2
import imutils
import numpy as np
from typing import Tuple

DATASET_DIR = './dataset'
CHARS_DIR = './chars'
char_counts = {}


def get_mod_thresh(captcha_image_file: str):
    image = cv2.imread(captcha_image_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # TODO: is it still necessary? Adds some extra padding around the image
    gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)

    # threshold the image (convert it to pure black and white)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # eroding/dilating it to remove noise
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    mod_thresh = cv2.erode(thresh, element, iterations=3)
    mod_thresh = cv2.dilate(mod_thresh, element, iterations=3)

    # Gives +10 finds
    kernel = np.ones((2, 2), np.uint8)
    mod_thresh = cv2.morphologyEx(mod_thresh, cv2.MORPH_CLOSE, kernel)

    return mod_thresh


def get_contours(mod_thresh):
    contours = cv2.findContours(mod_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Hack for compatibility with different OpenCV versions
    return contours[1] if imutils.is_cv3() else contours[0]


def get_letter_image_regions(contours):
    letter_image_regions = []

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        # Removing contours that are too small
        if w * h < 200:
            continue

        # Splitting regions too large into two. Not the best solution imo.
        if w / h > 1.25:
            half_width = int(w / 2)
            letter_image_regions.append((x, y, half_width, h))
            letter_image_regions.append((x + half_width, y, half_width, h))
        else:
            letter_image_regions.append((x, y, w, h))

    # Splitting the biggest one into two if there are three
    if len(letter_image_regions) == 3:
        letter_image_regions = sorted(letter_image_regions, key=lambda a: a[2])

        # Getting the one with the highest width from the back of the sorted list
        (x, y, w, h) = letter_image_regions.pop()

        # Splitting it in two
        letter_image_regions.append((x, y, w // 2, h))
        letter_image_regions.append((x + w // 2, y, w // 2, h))

    # TODO: if there are two, choose between splitting both into two or splitting the biggest one into three.

    # Sort the detected letter images based on the x coordinate to make sure
    # we are processing them from left-to-right so we match the right image
    # with the right letter
    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])
    return letter_image_regions


def save_chars_to_imgs(letter_image_regions: [Tuple[int]], captcha_correct_text: str, gray: [[]]) -> None:
    # Saving individual characters to images
    for char_box, char_text in zip(letter_image_regions, captcha_correct_text):
        # Grab the coordinates of the letter in the image
        x, y, w, h = char_box

        # Extract the letter from the original image with a 2-pixel margin around the edge
        letter_image = gray[y - 2:y + h + 2, x - 2:x + w + 2]

        if not letter_image.any():
            break

        # Get the folder to save the image in
        save_path = os.path.join(CHARS_DIR, char_text)

        # if the output directory does not exist, create it
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # write the letter image to a file
        count = char_counts.get(char_text, 1)
        p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
        print(p)
        cv2.imwrite(p, letter_image)

        # increment the count for the current key
        char_counts[char_text] = count + 1


def main():
    if os.path.exists(CHARS_DIR):
        shutil.rmtree(CHARS_DIR)
    os.makedirs(CHARS_DIR)

    nbr_valid = 0
    for filename in os.listdir(DATASET_DIR):
        captcha_image_file = os.path.join(DATASET_DIR, filename)

        mod_thresh = get_mod_thresh(captcha_image_file)
        contours = get_contours(mod_thresh)
        letter_image_regions = get_letter_image_regions(contours)

        # for char_box in letter_image_regions:
            # Grab the coordinates of the letter in the image
            # x, y, w, h = char_box

            # 2-pixel margin around the edge. For debug
            # cv2.rectangle(mod_thresh, (x - 2, y - 2), (x + w + 4, y + h + 4), (255, 255, 255), 1)

        if len(letter_image_regions) != 4:
            # cv2.imshow("Output", mod_thresh)
            # key = cv2.waitKey() & 0xFF
            # if key == ord("q"):
            #     exit()
            continue

        save_chars_to_imgs(letter_image_regions, os.path.splitext(filename)[0], gray)
        nbr_valid += 1

    print('{} images had four sections precisely.'.format(nbr_valid))


if __name__ == '__main__':
    main()
