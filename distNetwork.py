import tensorflow as tf
import tensorflow_datasets as tfds
import os
import numpy as np

datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True, data_dir='./data')

mnist_train, mnist_test = datasets['train'], datasets['test']
def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label


strategy = tf.distribute.MirroredStrategy()
num_train_examples = info.splits["train"].num_examples
num_test_examples = info.splits["test"].num_examples

BUFFER_SIZE = 10000

BATCH_SZIE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SZIE_PER_REPLICA * strategy.num_replicas_in_sync

train_dataset = mnist_train.map(scale).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
eval_dataset = mnist_test.map(scale).batch(BATCH_SIZE)



print("Number of devices: {0}".format(strategy.num_replicas_in_sync))

with strategy.scope():
    model = tf.keras.Sequential([tf.keras.layers.Conv2D(32, 3, activation="relu", input_shape=(28, 28, 1)),
                                 tf.keras.layers.MaxPooling2D(),
                                 tf.keras.layers.Flatten(),
                                 tf.keras.layers.Dense(64, activation="relu"),
                                 tf.keras.layers.Dense(10)
                                 ])

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"])

# model.fit(train_dataset, epochs=12)

os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"] = "1"

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images[..., None]
test_images = test_images[..., None]

BUFFER_SIZE = len(train_images)
BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE)
test_dataset = tf.data.Datset.from_tensor_slices((test_images, test_labels)).batch(GLOBAL_BATCH_SIZE)

train_dist_dataset = strategy.experimental_distrubute_dataset(train_dataset)
test_dist_dataset = strategy.experimental_distrubute_dataset(test_dataset)

EPOCHS = 10000
for epoch in range(EPOCHS):
    total_loss = 0.0
    num_batches = 0
    for batch in train_dist_dataset:
        total_loss += distributed_train_step(batch)
        num_batches += 1
    train_loss = train_loss / num_batches

@tf.function
def distributed_train_step(dataset_inputs):
    per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

def train_step(inputs):
    images, labels = inputs
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = compute_loss(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_accuracy.update_state(labels, predictions)
    return loss

@tf.function
def distributed_test_step(dataset_inputs):
  return strategy.run(test_step, args=(dataset_inputs,))

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    return model

with strategy.scope():

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

    def compute_loss(labels, predictions):
        per_example_loss = loss_object(labels, predictions)
        print(per_example_loss)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

    test_loss = tf.keras.metrics.Mean(name='test_loss')

    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    optimizer = tf.keras.optimizers.Adam()
    model = create_model()


def test_step(inputs):
    images, labels = inputs

    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss.update_state(t_loss)
    test_accuracy.update_state(labels, predictions)

EPOCHS = 10
for epoch in range(EPOCHS):
    total_loss = 0.0
    num_batches = 0
    for batch in train_dist_dataset:
        total_loss += distributed_train_step(batch)
        num_batches += 1
    train_loss = total_loss / num_batches

  # Do Testing
    for batch in test_dist_dataset:
        distributed_test_step(batch)

    template = ("Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, " "Test Accuracy: {}")

    print (template.format(epoch+1, train_loss, train_accuracy.result()*100, test_loss.result(), test_accuracy.result()*100))

    test_loss.reset_states()
    train_accuracy.reset_states()
    test_accuracy.reset_states()