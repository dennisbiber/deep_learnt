from tensorflow.keras.losses import Loss
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Lambda
from tensorflow.keras.optimizers import RMSprop
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.keras import backend as K
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf

class MyHuberLoss(Loss):

    def __init__(self, threshold):
        super().__init__()
        self._threshold = threshold

    def call(self, y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) <= self._threshold
        small_error_loss = tf.square(error) / 2
        big_error_loss = self._threshold * (tf.abs(error) - (0.5 * self._threshold))
        return tf.where(is_small_error, small_error_loss, big_error_loss)


def huberLoss(y_true, y_pred, threshold):
    error = y_true - y_pred
    is_small_error = tf.abs(error) <= threshold
    small_error_loss = tf.square(error) / 2
    big_error_loss = threshold * (tf.abs(error) - (0.5 * threshold))
    return tf.where(is_small_error, small_error_loss, big_error_loss)


def contrastive_loss_with_margin(margin):
    def contrastive_loss(y_true, y_pred):
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)
    return contrastive_loss

# TODO move below functions to another file

def main():

    # # inputs
    # xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)

    # # labels
    # ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

    # model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
    # model.compile(optimizer='sgd', loss=MyHuberLoss(threshold=1.02))
    # model.fit(xs, ys, epochs=500,verbose=0)
    # print(model.predict([10.0]))

    # Second Loss Function script
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # prepare train and test sets
    train_images = train_images.astype('float32')
    test_images = test_images.astype('float32')

    # normalize values
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # create pairs on train and test sets
    tr_pairs, tr_y = create_pairs_on_set(train_images, train_labels)
    ts_pairs, ts_y = create_pairs_on_set(test_images, test_labels)
    base_network = initialize_base_network()

    input_a = Input(shape=(28,28,), name="left_input")
    vect_output_a = base_network(input_a)

    # create the right input and point to the base network
    input_b = Input(shape=(28,28,), name="right_input")
    vect_output_b = base_network(input_b)

    # measure the similarity of the two vector outputs
    output = Lambda(euclidean_distance, name="output_layer", output_shape=eucl_dist_output_shape)([vect_output_a, vect_output_b])

    # specify the inputs and output of the model
    model = Model([input_a, input_b], output)

    # plot model graph
    plot_model(model, show_shapes=True, show_layer_names=True, to_file='outer-model.png')

    rms = RMSprop()
    model.compile(loss=contrastive_loss_with_margin(margin=1), optimizer=rms)
    history = model.fit([tr_pairs[:,0], tr_pairs[:,1]], tr_y, epochs=20, batch_size=128, validation_data=([ts_pairs[:,0], ts_pairs[:,1]], ts_y))

    loss = model.evaluate(x=[ts_pairs[:,0],ts_pairs[:,1]], y=ts_y)

    y_pred_train = model.predict([tr_pairs[:,0], tr_pairs[:,1]])
    train_accuracy = compute_accuracy(tr_y, y_pred_train)

    y_pred_test = model.predict([ts_pairs[:,0], ts_pairs[:,1]])
    test_accuracy = compute_accuracy(ts_y, y_pred_test)

    print("Loss = {}, Train Accuracy = {} Test Accuracy = {}".format(loss, train_accuracy, test_accuracy))
    plot_metrics(metric_name='loss', title="Loss", ylim=0.2, history=history)

    y_pred_train = np.squeeze(y_pred_train)
    indexes = np.random.choice(len(y_pred_train), size=10)
    display_images(tr_pairs[:, 0][indexes], tr_pairs[:, 1][indexes], y_pred_train[indexes], tr_y[indexes], "clothes and their dissimilarity", 10)


if __name__ == "__main__":
    main()