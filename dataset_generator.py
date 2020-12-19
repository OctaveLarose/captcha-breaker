import os
import shutil
from captcha.image import ImageCaptcha
import random
import string


def main():
    dataset_dir = './dataset'

    if os.path.isdir(dataset_dir):
        shutil.rmtree(dataset_dir)

    os.makedirs(dataset_dir)

    dataset_size = 1000

    for i in range(dataset_size):
        captcha_content = ''.join([random.choice(string.ascii_letters) for _ in range(4)])
        print('{}/{}: {}...'.format(i, dataset_size, captcha_content))

        img = ImageCaptcha()
        img_fn = os.path.join(dataset_dir, '{}.png'.format(captcha_content))
        img.write(captcha_content, img_fn)

    print('Done.')


if __name__ == "__main__":
    main()
