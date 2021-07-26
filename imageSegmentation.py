import tensorflow as tf
from tensorflow.keras.layers import UpSampling2D, Conv2DTranspose, Conv2D
from tensorflow.keras.layers import Add, Cropping2D, MaxPooling2D, Input
import numpy as np
import os
import zipfile
import PIL.Image, PIL.ImageFont, PIL.ImageDraw
from matplotlib import pyplot as plt
import tensorflow_datasets as tfds
import seaborn as sns


# x = UpSampling2D(size=(2,2),
#                data_format=None,
#                interpolation="nearest")(x)
# Conv2DTranspose(filters=32, kernal_size=(2,2))

# VDD 16 structure

class_names = ['sky', 'building','column/pole', 'road', 'side walk', 'vegetation', 'traffic light', 'fence', 'vehicle', 'pedestrian', 'byciclist', 'void']
BATCH_SIZE = 64

def block(x, n_convs, filters, kernel_size, activation, pool_size, pool_stride, block_name):
    for i in range(n_convs):
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size, activation=activation,
                   name="{}_conv{}".format(block_name, i + 1))(x)

    x = MaxPooling2D(pool_size=pool_size, strides=pool_stride,
                     name="{}_pool{}".format(block_name, i+1))(x)
    return x


def VGG_16(vgg_weights_path, image_input):
    x = block(image_input, n_convs=2, filters=64, kernel_size=(3,3),
              activation="relu", pool_size=(2, 2), pool_stride=(2, 2),
              block_name="block1")
    p1 = x
    x = block(x, n_convs=2, filters=128, kernel_size=(3, 3),
              activation="relu", pool_size=(2, 2), pool_stride=(2, 2),
              block_name="block2")

    p2 = x
    x = block(x, n_convs=3, filters=256, kernel_size=(3, 3), activation="relu",
              pool_size=(2, 2), pool_stride=(2, 2), block_name="block3")
    p3 = x
    x = block(x, n_convs=3, filters=512, kernel_size=(3, 3), activation="relu",
              pool_size=(2, 2), pool_stride=(2, 2), block_name="block4")
    p4 = x
    x = block(x, n_convs=3, filters=512, kernel_size=(3, 3), activation="relu",
              pool_size=(2, 2), pool_stride=(2, 2), block_name="block5")
    p5 = x

    vgg  = tf.keras.Model(image_input , p5)

    # load the pretrained weights you downloaded earlier
    vgg.load_weights(vgg_weights_path) 
    # number of filters for the output convolutional layers
    n = 4096

    # our input images are 224x224 pixels so they will be downsampled to 7x7 after the pooling layers above.
    # we can extract more features by chaining two more convolution layers.
    c6 = tf.keras.layers.Conv2D( n , ( 7 , 7 ) , activation='relu' , padding='same', name="conv6")(p5)
    c7 = tf.keras.layers.Conv2D( n , ( 1 , 1 ) , activation='relu' , padding='same', name="conv7")(c6)
    return (p1, p2, p3, p4 ,c7)


def fcn8_decoder(convs, n_classes):
    f1, f2, f3, f4, f5 = convs

    o = Conv2DTranspose(n_classes, kernel_size=(4, 4),
                        strides=(2, 2), use_bias=False)(f5)

    o = Cropping2D(cropping=(1, 1))(o)
    o2 = Conv2D(n_classes, (1,1),
                activation="relu", padding="same")(f4)

    print(o.shape)
    print(o2.shape)
    o = Add()([o, o2])
    o = Conv2DTranspose(n_classes, kernel_size=(4, 4),
                        strides=(2, 2))(o)

    o = Cropping2D(cropping=(1, 1))(o)
    o2 = Conv2D(n_classes, (1,1), activation="relu",
                padding="same")(f3)

    o = Add()([o, o2])
    o = Conv2DTranspose(n_classes, kernel_size=(8,8),
                        strides=(8,8))(o)

    o = Activation("softmax")(o)
    return o


def segmentation_model(vgg_weights_path):
    inputs = Input(shape=(224, 224, 3,))
    convs = VGG_16(vgg_weights_path, image_input=inputs)
    outputs = fcn8_decoder(convs, 12)
    model = Model(inputs=inputs, outputs=outputs)
    return model


def compute_metrics(y_true, y_pred, n_classes):
    class_wise_iou = []
    class_wise_dice_score = []

    smoothening_factor = 0.00001

    for i in range(n_classes):
        intersection = np.sum((y_pred == i) * (y_true == i))
        y_true_area = np.sum((y_true==i))
        y_pred_area = np.sum((y_pred == i))
        combined_area = y_true_area + y_pred_area
        union_area = combined_area - intersection
        iou = (intersection + smoothening_factor) / (union_area + smoothening_factor)
        class_wise_iou.append(iou)
        dice_score = 2 * (intersection + smoothening_factor) / (combined_area + smoothening_factor)
        class_wise_dice_score.append(dice_score)

    return class_wise_iou, class_wise_dice_score


def map_filename_to_image_and_mask(t_filename, a_filename, height=224, width=224):

    # Convert image and mask files to tensors 
    img_raw = tf.io.read_file(t_filename)
    anno_raw = tf.io.read_file(a_filename)
    image = tf.image.decode_jpeg(img_raw)
    annotation = tf.image.decode_jpeg(anno_raw)
 
    # Resize image and segmentation mask
    image = tf.image.resize(image, (height, width,))
    annotation = tf.image.resize(annotation, (height, width,))
    image = tf.reshape(image, (height, width, 3,))
    annotation = tf.cast(annotation, dtype=tf.int32)
    annotation = tf.reshape(annotation, (height, width, 1,))
    stack_list = []

    # Reshape segmentation masks
    for c in range(len(class_names)):
        mask = tf.equal(annotation[:,:,0], tf.constant(c))
        stack_list.append(tf.cast(mask, dtype=tf.int32))
  
    annotation = tf.stack(stack_list, axis=2)

    # Normalize pixels in the input image
    image = image/127.5
    image -= 1

    return image, annotation


def get_dataset_slice_paths(image_dir, label_map_dir):

    image_file_list = os.listdir(image_dir)
    label_map_file_list = os.listdir(label_map_dir)
    image_paths = [os.path.join(image_dir, fname) for fname in image_file_list]
    label_map_paths = [os.path.join(label_map_dir, fname) for fname in label_map_file_list]

    return image_paths, label_map_paths


def get_training_dataset(image_paths, label_map_paths):

    training_dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_map_paths))
    training_dataset = training_dataset.map(map_filename_to_image_and_mask)
    training_dataset = training_dataset.shuffle(100, reshuffle_each_iteration=True)
    training_dataset = training_dataset.batch(BATCH_SIZE)
    training_dataset = training_dataset.repeat()
    training_dataset = training_dataset.prefetch(-1)

    return training_dataset


def get_validation_dataset(image_paths, label_map_paths):

    validation_dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_map_paths))
    validation_dataset = validation_dataset.map(map_filename_to_image_and_mask)
    validation_dataset = validation_dataset.batch(BATCH_SIZE)
    validation_dataset = validation_dataset.repeat()  

    return validation_dataset


def fuse_with_pil(images):

    widths = (image.shape[1] for image in images)
    heights = (image.shape[0] for image in images)
    total_width = sum(widths)
    max_height = max(heights)

    new_im = PIL.Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        pil_image = PIL.Image.fromarray(np.uint8(im))
        new_im.paste(pil_image, (x_offset,0))
        x_offset += im.shape[1]
  
    return new_im


def give_color_to_annotation(annotation, colors):

    seg_img = np.zeros( (annotation.shape[0],annotation.shape[1], 3) ).astype('float')
  
    for c in range(12):
        segc = (annotation == c)
        seg_img[:,:,0] += segc*( colors[c][0] * 255.0)
        seg_img[:,:,1] += segc*( colors[c][1] * 255.0)
        seg_img[:,:,2] += segc*( colors[c][2] * 255.0)
  
    return seg_img


def show_predictions(image, labelmaps, titles, iou_list, dice_score_list, colors):

    true_img = give_color_to_annotation(labelmaps[1], colors)
    pred_img = give_color_to_annotation(labelmaps[0], colors)

    image = image + 1
    image = image * 127.5
    images = np.uint8([image, pred_img, true_img])

    metrics_by_id = [(idx, iou, dice_score) for idx, (iou, dice_score) in enumerate(zip(iou_list, dice_score_list)) if iou > 0.0]
    metrics_by_id.sort(key=lambda tup: tup[1], reverse=True)  # sorts in place
  
    display_string_list = ["{}: IOU: {} Dice Score: {}".format(class_names[idx], iou, dice_score) for idx, iou, dice_score in metrics_by_id]
    display_string = "\n\n".join(display_string_list) 

    plt.figure(figsize=(15, 4))

    for idx, im in enumerate(images):
        plt.subplot(1, 3, idx+1)
        if idx == 1:
            plt.xlabel(display_string)
        plt.xticks([])
        plt.yticks([])
        plt.title(titles[idx], fontsize=12)
        plt.imshow(im)


def show_annotation_and_image(image, annotation, colors):
  
    new_ann = np.argmax(annotation, axis=2)
    seg_img = give_color_to_annotation(new_ann, colors)
  
    image = image + 1
    image = image * 127.5
    image = np.uint8(image)
    images = [image, seg_img]
  
    images = [image, seg_img]
    fused_img = fuse_with_pil(images)
    plt.imshow(fused_img)


def list_show_annotation(dataset, colors):

    ds = dataset.unbatch()
    ds = ds.shuffle(buffer_size=100)

    plt.figure(figsize=(25, 15))
    plt.title("Images And Annotations")
    plt.subplots_adjust(bottom=0.1, top=0.9, hspace=0.05)

    for idx, (image, annotation) in enumerate(ds.take(9)):
        plt.subplot(3, 3, idx + 1)
        plt.yticks([])
        plt.xticks([])
        show_annotation_and_image(image.numpy(), annotation.numpy(), colors)


def get_images_and_segments_test_arrays():

    y_true_segments = []
    y_true_images = []
    test_count = 64

    ds = validation_dataset.unbatch()
    ds = ds.batch(101)

    for image, annotation in ds.take(1):
        y_true_images = image
        y_true_segments = annotation


    y_true_segments = y_true_segments[:test_count, : ,: , :]
    y_true_segments = np.argmax(y_true_segments, axis=3)  

    return y_true_images, y_true_segments


def main():

    local_zip = '/tmp/fcnn-dataset.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('/tmp/fcnn')
    zip_ref.close()

    training_image_paths, training_label_map_paths = get_dataset_slice_paths('/tmp/fcnn/dataset1/images_prepped_train/','/tmp/fcnn/dataset1/annotations_prepped_train/')
    validation_image_paths, validation_label_map_paths = get_dataset_slice_paths('/tmp/fcnn/dataset1/images_prepped_test/','/tmp/fcnn/dataset1/annotations_prepped_test/')

    # generate the train and val sets
    training_dataset = get_training_dataset(training_image_paths, training_label_map_paths)
    validation_dataset = get_validation_dataset(validation_image_paths, validation_label_map_paths)

    colors = sns.color_palette(None, len(class_names))

    # print class name - normalized RGB tuple pairs
    # the tuple values will be multiplied by 255 in the helper functions later
    # to convert to the (0,0,0) to (255,255,255) RGB values you might be familiar with
    for class_name, color in zip(class_names, colors):
        print(f'{class_name} -- {color}')

    list_show_annotation(training_dataset, colors)
    list_show_annotation(validation_dataset, colors)

    vgg_weights_path = "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
    model = segmentation_model(vgg_weights_path)
    model.summary()

    sgd = tf.keras.optimizers.SGD(lr=1E-2, momentum=0.9, nesterov=True)

    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    train_count = 367

    # number of validation images
    validation_count = 101

    EPOCHS = 170

    steps_per_epoch = train_count//BATCH_SIZE
    validation_steps = validation_count//BATCH_SIZE

    history = model.fit(training_dataset,
                        steps_per_epoch=steps_per_epoch, validation_data=validation_dataset, validation_steps=validation_steps, epochs=EPOCHS)

    y_true_images, y_true_segments = get_images_and_segments_test_arrays()
    results = model.predict(validation_dataset, steps=validation_steps)

    results = np.argmax(results, axis=3)
    integer_slider = 0

    # compute metrics
    iou, dice_score = compute_metrics(y_true_segments[integer_slider], results[integer_slider], 12)  

    # visualize the output and metrics
    show_predictions(y_true_images[integer_slider], [results[integer_slider], y_true_segments[integer_slider]], ["Image", "Predicted Mask", "True Mask"], iou, dice_score, colors)

    cls_wise_iou, cls_wise_dice_score = compute_metrics(y_true_segments, results, 12)
    for idx, iou in enumerate(cls_wise_iou):
        spaces = ' ' * (13-len(class_names[idx]) + 2)
        print("{}{}{} ".format(class_names[idx], spaces, iou)) 

    for idx, dice_score in enumerate(cls_wise_dice_score):
       spaces = ' ' * (13-len(class_names[idx]) + 2)
       print("{}{}{} ".format(class_names[idx], spaces, dice_score))


if __name__ == "__main__":
    main()