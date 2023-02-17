import tensorflow as tf
import timeit
from datetime import datetime


def regular_function(x, y, b):
    x = tf.matmul(x, y)
    x = x + b
    return x


graph_function = tf.function(regular_function)

x1 = tf.constant([[1.0, 2.0]])
y1 = tf.constant([[2.0], [3.0]])
b1 = tf.constant(4.0)

orig_value = regular_function(x1, y1, b1).numpy()
graph_value = graph_function(x1, y1, b1).numpy()
assert orig_value == graph_value


@tf.function
def outer_function(x):
    y = tf.constant([[2.0], [3.0]])
    b = tf.constant(4.0)
    return regular_function(x, y, b)


print(outer_function(tf.constant([[1.0, 2.0]])))


def simple_relu(x):
    if tf.greater(x, 0):
        return x
    else:
        return 0


tf_simple_relu = tf.function(simple_relu)

print("First branch, with graph:", tf_simple_relu(tf.constant(1)).numpy())
print("Second branch, with graph:", tf_simple_relu(tf.constant(-1)).numpy())


@tf.function
def my_relu(x):
    return tf.maximum(0.0, x)


# `my_relu` creates new graphs as it observes more signatures.
print(my_relu(tf.constant(5.5)))
print(my_relu([1, -1]))
print(my_relu(tf.constant([3.0, -3.0])))


@tf.function
def get_MSE(y_true, y_pred):
    print("Calculating MSE!")
    sq_diff = tf.pow(y_true - y_pred, 2)
    return tf.reduce_mean(sq_diff)


y_true = tf.random.uniform([5], maxval=10, dtype=tf.int32)
y_pred = tf.random.uniform([5], maxval=10, dtype=tf.int32)

print(y_true)
print(y_pred)

error = get_MSE(y_true, y_pred)
error = get_MSE(y_true, y_pred)
error = get_MSE(y_true, y_pred)
