import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import matplotlib.pyplot as plt

import numpy as np


@tf.function
def map_fn(img, label):
    image_height = 224
    image_width = 224
    # resize the image
    img = tf.image.resize(img, (image_height, image_width))
    # normalize the image
    img /= 255.
    return img, label

def prepare_dataset(train_examples, validation_examples, test_examples, num_examples, map_fn, batch_size):
    train_ds = train_examples.map(map_fn).shuffle(128).batch(batch_size)
    valid_ds = validation_examples.map(map_fn).batch(batch_size)
    test_ds = test_examples.map(map_fn).batch(batch_size)
    return train_ds, valid_ds, test_ds


def set_adam_optimizer():
    return tf.keras.optimizers.Adam()


def set_sparse_cat_crossentropy_loss():
    # Define object oriented metric of Sparse categorical crossentropy for train and val loss
    train_loss = tf.keras.losses.SparseCategoricalCrossentropy()
    val_loss = tf.keras.losses.SparseCategoricalCrossentropy()
    return train_loss, val_loss


def set_sparse_cat_crossentropy_accuracy():
    # Define object oriented metric of Sparse categorical accuracy for train and val accuracy
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    return train_accuracy, val_accuracy

def train_one_step(model, optimizer, x, y, train_loss, train_accuracy):
    '''    
    Args:
        model (keras Model) -- image classifier
        optimizer (keras Optimizer) -- optimizer to use during training
        x (Tensor) -- training images
        y (Tensor) -- training labels
        train_loss (keras Loss) -- loss object for training
        train_accuracy (keras Metric) -- accuracy metric for training
    '''
    with tf.GradientTape() as tape:
        # Run the model on input x to get predictions
        predictions = model(x)
        # Compute the training loss using `train_loss`, passing in the true y and the predicted y
        loss = train_loss(y, predictions)

    # Using the tape and loss, compute the gradients on model variables using tape.gradient
    grads = tape.gradient(loss, model.trainable_weights)
    
    # Zip the gradients and model variables, and then apply the result on the optimizer
    optimizer.apply_gradients(zip(grads , model.trainable_weights))

    # Call the train accuracy object on ground truth and predictions
    train_accuracy(y , predictions)
    return loss


@tf.function
def train(model, optimizer, epochs, device, train_ds, train_loss, train_accuracy, valid_ds, val_loss, val_accuracy):
    '''    
    Args:
        model (keras Model) -- image classifier
        optimizer (keras Optimizer) -- optimizer to use during training
        epochs (int) -- number of epochs
        train_ds (tf Dataset) -- the train set containing image-label pairs
        train_loss (keras Loss) -- loss function for training
        train_accuracy (keras Metric) -- accuracy metric for training
        valid_ds (Tensor) -- the val set containing image-label pairs
        val_loss (keras Loss) -- loss object for validation
        val_accuracy (keras Metric) -- accuracy metric for validation
    '''
    step = 0
    loss = 0.0
    for epoch in range(epochs):
        for x, y in train_ds:
            # training step number increments at each iteration
            step += 1
            with tf.device(device_name=device):
                # Run one training step by passing appropriate model parameters
                # required by the function and finally get the loss to report the results
                loss = train_one_step(model, optimizer, x, y, train_loss, train_accuracy)
            # Use tf.print to report your results.
            # Print the training step number, loss and accuracy
            tf.print('Step', step, 
                   ': train loss', loss, 
                   '; train accuracy', train_accuracy.result())

        with tf.device(device_name=device):
            for x, y in valid_ds:
                # Call the model on the batches of inputs x and get the predictions
                y_pred = model(x)
                loss = val_loss(y, y_pred)
                val_accuracy(y, y_pred)
        
        # Print the validation loss and accuracy
        tf.print('val loss', loss, '; val accuracy', val_accuracy)


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    img = np.squeeze(img)

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    
    # green-colored annotations will mark correct predictions. red otherwise.
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'
    
    # print the true label first
    print(true_label)
  
    # show the image and overlay the prediction
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)


def main():
	splits, info = tfds.load('horses_or_humans', as_supervised=True, with_info=True, split=['train[:80%]', 'train[80%:]', 'test'], data_dir='./data')

	(train_examples, validation_examples, test_examples) = splits

	num_examples = info.splits['train'].num_examples
	num_classes = info.features['label'].num_classes

	BATCH_SIZE = 32
	IMAGE_SIZE = 224

	test_image, test_label = list(train_examples)[0]

	test_result = map_fn(test_image, test_label)

	train_ds, valid_ds, test_ds = prepare_dataset(train_examples, validation_examples, 
												  test_examples, num_examples, map_fn, 
												  BATCH_SIZE)
	MODULE_HANDLE = 'data/resnet_50_feature_vector'
	model = tf.keras.Sequential([
	    hub.KerasLayer(MODULE_HANDLE, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
	    tf.keras.layers.Dense(num_classes, activation='softmax')
	])
	model.summary()

	optimizer = set_adam_optimizer()
	train_loss, val_loss = set_sparse_cat_crossentropy_loss()
	train_accuracy, val_accuracy = set_sparse_cat_crossentropy_accuracy()
	device = '/gpu:0' if tf.config.list_physical_devices('GPU') else '/cpu:0'
	EPOCHS = 2

	train(model, optimizer, EPOCHS, device, train_ds, train_loss, 
		  train_accuracy, valid_ds, val_loss, val_accuracy)

	test_imgs = []
	test_labels = []

	predictions = []
	with tf.device(device_name=device):
	    for images, labels in test_ds:
	        preds = model(images)
	        preds = preds.numpy()
	        predictions.extend(preds)

	        test_imgs.extend(images.numpy())
	        test_labels.extend(labels.numpy())

	class_names = ['horse', 'human']

	index = 8 
	plt.figure(figsize=(6,3))
	plt.subplot(1,2,1)
	plot_image(index, predictions, test_labels, test_imgs)
	plt.show()

if __name__ == "__main__":
	main()