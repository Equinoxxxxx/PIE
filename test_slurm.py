import tensorflow as tf
import time
# import torch

def main():
    print('Hello, world!')
    time.sleep(15)
    print('GPU available (from tensorflow)', tf.test.is_gpu_available())
    # print('GPU available (from torch)', torch.cuda.is_available(), torch.cuda.get_device_name(0))

if __name__ == '__main__':
    main()