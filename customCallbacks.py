import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

import datetime
import io

from PIL import Image
from IPython.display import Image as IPyImage
import imageio
tf.get_logger().setLevel('INFO')

class CustomCallback(tf.keras.callbacks.Callback):

    def on_train_batch_begin(self, batch, logs=None):
        print('Training: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))

    def on_train_batch_end(self, batch, logs=None):
        print('Training: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))


class DetectOverfittingCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold=0.7):
        super(DetectOverfittingCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        ratio = logs["val_loss"] / logs["loss"]
        print("Epoch: {}, Val/Train loss ratio: {:.2f}".format(epoch, ratio))

        if ratio > self.threshold:
            print("Stopping training...")
            self.model.stop_training = True


class VisCallback(tf.keras.callbacks.Callback):
    def __init__(self, inputs, ground_truth, GIF_PATH, display_freq=10, n_samples=10):
        self.inputs = inputs
        self.ground_truth = ground_truth
        self.images = []
        self.display_freq = display_freq
        self.n_samples = n_samples
        self.GIF_PATH = GIF_PATH

    def on_epoch_end(self, epoch, logs=None):
        # Randomly sample data
        indexes = np.random.choice(len(self.inputs), size=self.n_samples)
        X_test, y_test = self.inputs[indexes], self.ground_truth[indexes]
        predictions = np.argmax(self.model.predict(X_test), axis=1)

        # Plot the digits
        display_digits(X_test, predictions, y_test, epoch, n=self.display_freq)

        # Save the figure
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        self.images.append(np.array(image))

        # Display the digits every 'display_freq' number of epochs
        if epoch % self.display_freq == 0:
            plt.show()

    def on_train_end(self, logs=None):
        imageio.mimsave(self.GIF_PATH, self.images, fps=1)


def get_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(32, activation='linear', input_dim=784))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def display_digits(inputs, outputs, ground_truth, epoch, n=10):
    plt.clf()

    plt.yticks([])
    plt.grid(None)
    inputs = np.reshape(inputs, [n, 28, 28])
    inputs = np.swapaxes(inputs, 0, 1)
    inputs = np.reshape(inputs, [28, 28*n])
    plt.imshow(inputs)
    plt.xticks([28*x+14 for x in range(n)], outputs)
    for i,t in enumerate(plt.gca().xaxis.get_ticklabels()):
        if outputs[i] == ground_truth[i]: 
            t.set_color('green') 
        else: 
            t.set_color('red')
    plt.grid(None)




def main():
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
	x_train = x_train.reshape(60000, 784).astype('float32') / 255
	x_test = x_test.reshape(10000, 784).astype('float32') / 255

	model = get_model()
	_ = model.fit(x_train, y_train,
	          batch_size=64,
	          epochs=1,
	          steps_per_epoch=5,
	          verbose=0,
	          callbacks=[CustomCallback()])

	callback = tf.keras.callbacks.LambdaCallback(
	    on_epoch_end=lambda epoch,logs: 
	    print("Epoch: {}, Val/Train loss ratio: {:.2f}".format(epoch, logs["val_loss"] / logs["loss"]))
	)

	model = get_model()
	_ = model.fit(x_train, y_train,
	          validation_data=(x_test, y_test),
	          batch_size=64,
	          epochs=3,
	          verbose=0,
	          callbacks=[callback])

	model = get_model()
	_ = model.fit(x_train, y_train,
	              validation_data=(x_test, y_test),
	              batch_size=64,
	              epochs=3,
	              verbose=0,
	              callbacks=[DetectOverfittingCallback()])

	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
	x_train = x_train.reshape(60000, 784).astype('float32') / 255
	x_test = x_test.reshape(10000, 784).astype('float32') / 255

	plt.rc('font', size=20)
	plt.rc('figure', figsize=(15, 3))

	GIF_PATH = './animation.gif'

	model = get_model()
	model.fit(x_train, y_train,
	          batch_size=64,
	          epochs=20,
	          verbose=0,
	          callbacks=[VisCallback(x_test, y_test, GIF_PATH)])

	SCALE = 60

	# FYI, the format is set to PNG here to bypass checks for acceptable embeddings
	IPyImage(GIF_PATH, format='png', width=15 * SCALE, height=3 * SCALE) 

if __name__ == "__main__":
	main()