import tensorflow as tf
import numpy as np
import time

if __name__ == "__main__":
    """
    # NumPy 版
    a_np = np.random.randn(5000,5000)
    b_np = np.random.randn(5000,5000)

    t = time.time()
    c_np = a_np @ b_np
    print("NumPy:", time.time() - t)
    """
    # TensorFlow GPU 版
    a_tf = tf.random.normal([5000,5000])
    b_tf = tf.random.normal([5000,5000])

    t = time.time()
    c_tf = tf.matmul(a_tf, b_tf)
    print("TF GPU:", time.time() - t)