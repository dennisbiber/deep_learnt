from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, concatenate, Activation, MaxPool2D
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, Add, GlobalAveragePooling2D
import tensorflow_datasets as tfds
import tensorflow as tf


input_a = Input(shape=[1], name="Wide_Input")
input_b = Input(shape=[1], name="Deep_Input")
hidden_1 = Dense(30, activation="relu")(input_b)
hidden_2 = Dense(30, activation="relu")(hidden_1)

concat = concatenate([input_a, hidden_2])
output = Dense(1, name="Output")(concat)
aux_output = Dense(1, name="aux_Output")(hidden_2)
model = Model(inputs=[input_a, input_b], outputs=[output, aux_output])

class WideAndDeepModel(Model):
    def __init__(self, units=30, activation='relu', **kwargs):
        '''initializes the instance attributes'''
        super().__init__(**kwargs)
        self.hidden1 = Dense(units, activation=activation)
        self.hidden2 = Dense(units, activation=activation)
        self.main_output = Dense(1)
        self.aux_output = Dense(1)

    def call(self, inputs):
        '''defines the network architecture'''
        input_A, input_B = inputs
        hidden1 = self.hidden1(input_B)
        hidden2 = self.hidden2(hidden1)
        concat = concatenate([input_A, hidden2])
        main_output = self.main_output(concat)
        aux_output = self.aux_output(hidden2)
        
        return main_output, aux_output


class CNNResidual(Layer):
	def __init__(self, layers, filters, **kwargs):
		super().__init__(**kwargs)
		self.hidden = [Conv2D(filters, (3,3), activation="relu") for _ in range(layers)]

	def call(self, inputs):
		x = inputs
		for layer in self.hidden:
			x = layer(x)
		return inputs + x

class DNNResidual(Layer):
	def __init__(self, layers, neurons, **kwargs):
		super().__init__(**kwargs)
		self.hidden = [Dense(neurons, activation="relu") for _ in range(layers)]

	def call(self, inputs):
		x = inputs
		for layer in self.hidden:
			x = layer(x)
		return inputs + x

class MyResidual(Model):
	def __init__(self, **kwargs):
		super(MyResidual, self).__init__(**kwargs)
		self.hidden1 = Dense(30, activation="relu")
		self.block1 = CNNResidual(2, 32)
		self.block2 = DNNResidual(2, 64)
		self.out = Dense(1)

	def call(self, inputs):
		x = self.hidden1(inputs)
		x = self.block1(x)
		for _ in range(1, 4):
			x = self.block2(x)
		return self.out(x)

class IdentityBlock(Model):
	def __init__(self, filters, kernel_size):
		super(IdentityBlock, self).__init__(name='')
		self.conv1 = Conv2D(filters, kernel_size, padding="same")
		self.bn1 = BatchNormalization()

		self.conv2 = Conv2D(filters, kernel_size, padding="same")
		self.bn2 = BatchNormalization()

		self.act = Activation("relu")
		self.add = Add()

	def call(self, input_tensor):
		x = self.conv1(input_tensor)
		x = self.bn1(x)
		x = self.act(x)

		x = self.conv2(x)
		x = self.bn2(x)
		x = self.act(x)

		x = self.add([x, input_tensor])
		x = self.act(x)
		return x


class ResNet(Model):
	def __init__(self, num_classes):
		super(ResNet, self).__init__()
		self.conv = Conv2D(64, 7, padding="same")
		self.bn = BatchNormalization()
		self.act = Activation("relu")
		self.max_pool = MaxPool2D((3,3))
		self.id1a = IdentityBlock(64, 3)
		self.id1b = IdentityBlock(64, 3)
		self.global_pool = GlobalAveragePooling2D()
		self.classifier = Dense(num_classes, activation="softmax")

	def call(self, inputs):
		x = self.conv(inputs)
		x = self.bn(x)
		x = self.act(x)
		x = self.max_pool(x)

		x = self.id1a(x)
		x =self.id1b(x)

		x = self.global_pool(x)
		return self.classifier(x)\

def preprocess(features):
    return tf.cast(features['image'], tf.float32) / 255., features['label']

model = WideAndDeepModel()

model = MyResidual()

resnet = ResNet(10)
resnet.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
			   metrics=["accuracy"])
dataset = tfds.load('mnist', split=tfds.Split.TRAIN)
dataset = dataset.map(preprocess).batch(32)
resnet.fit(dataset, epochs=1)
