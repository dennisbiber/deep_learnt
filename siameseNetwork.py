from lossFunctions import *
from tensorflow.keras.datasets import fashion_mnist

__author__ = "Dennis Biber <dennisbiber88@gmail.com>"

class PrepModel(object):

    def __init__(self):
        super().__init__()

    def create_pairs(self, x, digit_indices):
        '''Positive and negative pair creation.
        Alternates between positive and negative pairs.
        '''
        pairs = []
        labels = []
        n = min([len(digit_indices[d]) for d in range(10)]) - 1
        
        for d in range(10):
            for i in range(n):
                z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
                pairs += [[x[z1], x[z2]]]
                inc = random.randrange(1, 10)
                dn = (d + inc) % 10
                z1, z2 = digit_indices[d][i], digit_indices[dn][i]
                pairs += [[x[z1], x[z2]]]
                labels += [1, 0]
                
        return np.array(pairs), np.array(labels)


    def create_pairs_on_set(self, images, labels):
        
        digit_indices = [np.where(labels == i)[0] for i in range(10)]
        pairs, y = self.create_pairs(images, digit_indices)
        y = y.astype('float32')
        
        return pairs, y


    def initialize_base_network(self):
        input = Input(shape=(28,28,), name="base_input")
        x = Flatten(name="flatten_input")(input)
        x = Dense(128, activation='relu', name="first_base_dense")(x)
        x = Dropout(0.1, name="first_dropout")(x)
        x = Dense(128, activation='relu', name="second_base_dense")(x)
        x = Dropout(0.1, name="second_dropout")(x)
        x = Dense(128, activation='relu', name="third_base_dense")(x)

        return Model(inputs=input, outputs=x)


    def euclidean_distance(self, vects):
        x, y = vects
        sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
        return K.sqrt(K.maximum(sum_square, K.epsilon()))


    def eucl_dist_output_shape(self, shapes):
        shape1, shape2 = shapes
        return (shape1[0], 1)

    def compute_accuracy(self, y_true, y_pred):
        '''Compute classification accuracy with a fixed threshold on distances.
        '''
        pred = y_pred.ravel() < 0.5
        return np.mean(pred == y_true)


class Plots(object):

    def __init__(self):
        super().__init__()

    def plot_metrics(self, metric_name, title, ylim=5, history=None):
        plt.title(title)
        plt.ylim(0,ylim)
        plt.plot(history.history[metric_name],color='blue',label=metric_name)
        plt.plot(history.history['val_' + metric_name],color='green',label='val_' + metric_name)

    def visualize_images(self):
        plt.rc('image', cmap='gray_r')
        plt.rc('grid', linewidth=0)
        plt.rc('xtick', top=False, bottom=False, labelsize='large')
        plt.rc('ytick', left=False, right=False, labelsize='large')
        plt.rc('axes', facecolor='F8F8F8', titlesize="large", edgecolor='white')
        plt.rc('text', color='a8151a')
        plt.rc('figure', facecolor='F0F0F0')# Matplotlib fonts


    # utility to display a row of digits with their predictions
    def display_images(self, left, right, predictions, labels, title, n):
        plt.figure(figsize=(17,3))
        plt.title(title)
        plt.yticks([])
        plt.xticks([])
        plt.grid(None)
        left = np.reshape(left, [n, 28, 28])
        left = np.swapaxes(left, 0, 1)
        left = np.reshape(left, [28, 28*n])
        plt.imshow(left)
        plt.figure(figsize=(17,3))
        plt.yticks([])
        plt.xticks([28*x+14 for x in range(n)], predictions)
        for i,t in enumerate(plt.gca().xaxis.get_ticklabels()):
            if predictions[i] > 0.5: t.set_color('red') # bad predictions in red
        plt.grid(None)
        right = np.reshape(right, [n, 28, 28])
        right = np.swapaxes(right, 0, 1)
        right = np.reshape(right, [28, 28*n])
        plt.imshow(right)

def main():
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # prepare train and test sets
    train_images = train_images.astype('float32')
    test_images = test_images.astype('float32')

    # normalize values
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # create pairs on train and test sets
    PM = PrepModel()
    tr_pairs, tr_y = PM.create_pairs_on_set(train_images, train_labels)
    ts_pairs, ts_y = PM.create_pairs_on_set(test_images, test_labels)
    base_network = PM.initialize_base_network()

    input_a = Input(shape=(28,28,), name="left_input")
    vect_output_a = base_network(input_a)

    # create the right input and point to the base network
    input_b = Input(shape=(28,28,), name="right_input")
    vect_output_b = base_network(input_b)

    # measure the similarity of the two vector outputs
    output = Lambda(PM.euclidean_distance, name="output_layer", output_shape=PM.eucl_dist_output_shape)([vect_output_a, vect_output_b])

    # specify the inputs and output of the model
    model = Model([input_a, input_b], output)

    # plot model graph
    plot_model(model, show_shapes=True, show_layer_names=True, to_file='outer-model.png')

    rms = RMSprop()
    model.compile(loss=contrastive_loss_with_margin(margin=1), optimizer=rms)
    history = model.fit([tr_pairs[:,0], tr_pairs[:,1]], tr_y, epochs=20, batch_size=128, validation_data=([ts_pairs[:,0], ts_pairs[:,1]], ts_y))

    loss = model.evaluate(x=[ts_pairs[:,0],ts_pairs[:,1]], y=ts_y)

    y_pred_train = model.predict([tr_pairs[:,0], tr_pairs[:,1]])
    train_accuracy = PM.compute_accuracy(tr_y, y_pred_train)

    y_pred_test = model.predict([ts_pairs[:,0], ts_pairs[:,1]])
    test_accuracy = PM.compute_accuracy(ts_y, y_pred_test)

    print("Loss = {}, Train Accuracy = {} Test Accuracy = {}".format(loss, train_accuracy, test_accuracy))
    _plot = Plots()
    _plot.plot_metrics(metric_name='loss', title="Loss", ylim=0.2, history=history)

    y_pred_train = np.squeeze(y_pred_train)
    indexes = np.random.choice(len(y_pred_train), size=10)
    _plot.display_images(tr_pairs[:, 0][indexes], tr_pairs[:, 1][indexes], y_pred_train[indexes], tr_y[indexes], "clothes and their dissimilarity", 10)


if __name__ == "__main__":
    main()