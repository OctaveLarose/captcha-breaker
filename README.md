# captcha-breaker
Breaking simple CAPTCHAs for an assignment in my Computer Security module.

I took it as an opportunity to better myself with OpenCV and Tensorflow.

The way it works is that it extracts individual characters from generated CAPTCHA images, 
trains a neural network to recognize them, then uses that same NN to name every character in the CAPTCHAs.

# Requirements
*Python 3.8*. 

Untested with other versions of Python 3. Tensorflow unavailable on the newest version, 3.9,
as of time of writing.

# Usage

```
virtualenv . ; source venv/bin/activate ; pip install -r requirements.txt
python dataset_generator.py
python extract_characters.py
python neural_net_black_magic.py
python solve_captchas.py
```

1. (Optional) Setting up your virtualenv via `virtualenv . ; venv/bin/pip install -r requirements.txt`

2. Generating the CAPTCHA dataset via `python dataset_generator.py`.

3. Extracting individual characters from the CAPTCHA dataset thanks to `python extract_characters.py`.

4. Training the neural network with `python neural_net_black_magic.py`.

5. Actually solving CAPTCHAs using `python solve_captchas.py`!