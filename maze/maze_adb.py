import numpy as np
import subprocess


def get_screen():
    pipe = subprocess.Popen("adb exec-out screencap -p",
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, shell=True)
    image_bytes = pipe.stdout.read()
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    return image_array


def swipe():
    pipe = subprocess.Popen("adb shell input swipe 678 411 700 411",
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, shell=True)


swipe()
