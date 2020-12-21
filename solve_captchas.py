from keras.models import load_model
from extract_characters import get_letter_image_regions, get_mod_imgs, get_contours
from helpers import resize_to_fit
from imutils import paths
import numpy as np
import cv2
import pickle


MODEL_FILENAME = "captcha_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"
DATASET_DIR = "./dataset"


nb_contour_errors = 0
nb_resize_errors = 0
nb_invalid_guesses = 0
nb_valid_guesses = 0

# Load up the model labels (so we can translate model predictions to actual letters)
with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)

# Load the trained neural network
model = load_model(MODEL_FILENAME)

# Grab some random CAPTCHA images to test against.
# In the real world, you'd replace this section with code to grab a real
# CAPTCHA image from a live website.
captcha_image_files = list(paths.list_images(DATASET_DIR))
# captcha_image_files = np.random.choice(captcha_image_files, size=(10,), replace=False)

# loop over the image paths
for captcha_image_file in captcha_image_files:
    image, _ = get_mod_imgs(captcha_image_file)
    contours = get_contours(image)
    letter_image_regions = get_letter_image_regions(contours)

    # If we found more or less than 4 letters in the captcha, our letter extraction
    # didn't work correcly. Skip the image instead of saving bad training data!
    if len(letter_image_regions) != 4:
        print('Couldn\'t capture regions correctly.')
        nb_contour_errors += 1
        continue

    # Create an output image and a list to hold our predicted letters
    output = cv2.merge([image] * 3)
    predictions = []

    try:
        # loop over the lektters
        for letter_bounding_box in letter_image_regions:
            # Grab the coordinates of the letter in the image
            x, y, w, h = letter_bounding_box

            # Extract the letter from the original image with a 2-pixel margin around the edge
            letter_image = image[y - 2:y + h + 2, x - 2:x + w + 2]

            # Re-size the letter image to 20x20 pixels to match training data
            letter_image = resize_to_fit(letter_image, 20, 20)

            # Turn the single image into a 4d list of images to make Keras happy
            letter_image = np.expand_dims(letter_image, axis=2)
            letter_image = np.expand_dims(letter_image, axis=0)

            # Ask the neural network to make a prediction
            prediction = model.predict(letter_image)

            # Convert the one-hot-encoded prediction back to a normal letter
            letter = lb.inverse_transform(prediction)[0]
            predictions.append(letter)

            # draw the prediction on the output image
            cv2.rectangle(output, (x - 2, y - 2), (x + w + 4, y + h + 4), (0, 255, 0), 1)
            cv2.putText(output, letter, (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    except cv2.error:
        print('cv2 error when resizing.')
        nb_resize_errors += 1
        continue

    # Print the captcha's text
    predicted_captcha_text = "".join(predictions)
    actual_captcha_text = captcha_image_file[-8:-4]
    print("Predicted CAPTCHA text is: {} (Actual: {})".format(predicted_captcha_text, actual_captcha_text))

    if predicted_captcha_text == actual_captcha_text:
        nb_valid_guesses += 1
    else:
        nb_invalid_guesses += 1

    # Show the annotated image
    # cv2.imshow("Output", output)
    # key = cv2.waitKey() & 0xFF
    # if key == ord("q"):
    #      exit()

print("\nFinal results (on {} elements):".format(len(captcha_image_files)))
print("Number of contour related errors (not 4 distinct characters found) : {}".format(nb_contour_errors))
print("Number of CV2 resizing errors (couldn't resize image, so couldn't feed it to the NN) : {}".format(nb_resize_errors))
print("Number of invalid guesses : {}".format(nb_invalid_guesses))
print("Number of valid guesses : {}".format(nb_valid_guesses))

print("Valid/Invalid guesses ratio (including errors): {}/{} = {}%".format(nb_valid_guesses,
                                                                           nb_invalid_guesses + nb_contour_errors + nb_resize_errors,
                                                                           nb_valid_guesses / 1000))

print("Valid/Invalid guesses ratio (excluding errors): {}/{} = {}%".format(nb_valid_guesses,
                                                                           nb_invalid_guesses,
                                                                           nb_valid_guesses / (1000 - nb_contour_errors - nb_resize_errors)))