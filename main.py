import os.path
import argparse
import numpy as np
import scipy.misc
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

Epochs = 20
Batch_Size = 2
Learning_Rate = 0.0001
Dropout = 0.7
num_classes = 2
image_shape = (160, 576)
data_dir = './data'
runs_dir = './runs'
models_dir = './models'
log_dir = './logs'

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    # ENVODER PART

    # Get Tensors to be returned from graph
    graph = tf.get_default_graph()

    input_image = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)

    # Layers 3 and 4 are used to add skip connections to the corresponding transpose layers
    l3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    l4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)

    # Add a 1x1 convolutional layer to layer 7
    l7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return input_image, keep_prob, l3, l4, l7

tests.test_load_vgg(load_vgg, tf)


def custom_init(mean=0.0, stddev=0.01, seed=0):
    return tf.truncated_normal_initializer(mean=mean, stddev=stddev, seed=seed)

def custom_regularizer(scale=1e-3):
    return tf.contrib.layers.l2_regularizer(scale=scale)

def conv_1x1(layer, num_outputs, layer_name):
    with tf.name_scope("conv_1x1"):
        return tf.layers.conv2d(layer,
                                filters=num_outputs,
                                kernel_size=(1,1),
                                strides=(1,1),
                                padding='same',
                                kernel_initializer=custom_init(),
                                kernel_regularizer=custom_regularizer(),
                                name=layer_name)

def upsample(layer, num_outputs, kernel_size, strides, layer_name):
    with tf.name_scope("deconv"):
        return tf.layers.conv2d_transpose(layer,
                                          filters=num_outputs,
                                          kernel_size=kernel_size,
                                          strides=strides,
                                          padding='same',
                                          kernel_initializer=custom_init(),
                                          kernel_regularizer=custom_regularizer(),
                                          name=layer_name)

def skip_connection(layer1, layer2, name):
    with tf.name_scope("skip_connection"):
        return tf.add(layer1, layer2, name=name)

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    # NOTE: How to create a 1x1 convolution was explained in detail in Scene Understanding, Lesson 10.7
    conv7_1x1 = conv_1x1(vgg_layer7_out, num_classes, "layer_7_1x1")
    conv4_1x1 = conv_1x1(vgg_layer4_out, num_classes, "layer_4_1x1")
    conv3_1x1 = conv_1x1(vgg_layer3_out, num_classes, "layer_3_1x1")

    #####################################################################################################

    # DECODER
    # Upsample until the output has the same width and height as the input images.
    # NOTE: The kernel and stride sizes are taken from Scene Understanding, Lesson 10.8
    deconv1 = upsample(conv7_1x1, num_classes, (4,4), (2,2), "deconv1")
    deconv2 = skip_connection(deconv1, conv4_1x1, "deconv2")
    deconv3 = upsample(deconv2, num_classes, (4,4), (2,2), "deconv3")
    deconv4 = skip_connection(deconv3, conv3_1x1, "deconv4")
    deconv_output = upsample(deconv4, num_classes, 16, 8, "deconv_output")

    return deconv_output

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    # NOTE: How to implement this part is described in detail in Scene Understanding, Lesson 10.9

    # Reshape 4D tensors to 2D, each row represents a pixel, each column a class
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    class_labels = tf.reshape(correct_label, (-1, num_classes))

    with tf.name_scope("cross_entropy"):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=class_labels)
        cross_entropy_loss = tf.reduce_mean(cross_entropy)
        tf.summary.scalar("cross_entropy", cross_entropy_loss)

    with tf.name_scope("optimizer"):
        # Adam algorithm is used as optimizer. It tries to minimize the "cross_entropy_loss" function
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss

tests.test_optimize(optimize)

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate,
             summary=None, writer=None):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    counter = 1
    for epoch in range(epochs):
        losses = []

        for images, labels in get_batches_fn(batch_size):
            feed = {input_image: images,
                    correct_label: labels,
                    keep_prob: Dropout,
                    learning_rate: Learning_Rate}

            if summary != None:
                _, loss, s = sess.run([train_op, cross_entropy_loss, summary], feed_dict=feed)

                if counter % 5 == 0:
                    writer.add_summary(s, counter)
            else:
                _, loss = sess.run([train_op, cross_entropy_loss], feed_dict=feed)

            losses.append(loss)

            print("--> Run: ", counter, " loss:", loss)

            counter += 1

        total_loss = sum(losses) / len(losses)

        print()
        print("Epoch: ", epoch + 1, " of ", Epochs, "total loss: ", total_loss)
        print()

tests.test_train_nn(train_nn)

def mean_iou(ground_truth, prediction, num_classes):
    with tf.name_scope("MeanIoU"):
        iou, iou_op = tf.metrics.mean_iou(ground_truth, prediction, num_classes)
        tf.summary.scalar("mean_iou", iou)
        return iou, iou_op

def evaluate_nn(sess, batch_size, get_batches_fn, keep_prob, logits,
                input_image, prediction, ground_truth, iou, iou_op,
                summary, writer, out_dir='evaluations'):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of testing data.  Call using get_batches_fn(batch_size)
    :param keep_prob: TF Placeholder for dropout keep probability
    :param logits:
    :param input_image: TF Placeholder for input images
    :param prediction:
    :param ground_truth:
    :param iou:
    :param iou_op:
    :param out_dir:
    """

    mean_ious = []
    counter = 1
    img_counter = 1
    dropout = 1.0

    for images, gt_images in get_batches_fn(batch_size):
        for i in range(len(images)):
            image = images[i]
            gt_image = gt_images[i]

            feed = {keep_prob: dropout, input_image: [image]}
            pixel_probs = sess.run([tf.nn.softmax(logits)], feed_dict=feed)
            pixel_probs = pixel_probs[0][:, 1].reshape(image_shape[0], image_shape[1])

            predicted_labels = (pixel_probs > 0.5).reshape(image_shape[0], image_shape[1], 1)
            predicted_image = np.concatenate((np.invert(predicted_labels), predicted_labels), axis=2)

            feed = {
                ground_truth : [gt_image],
                prediction : [predicted_image],
                input_image: [image],
                keep_prob: dropout
            }

            sess.run(iou_op, feed_dict=feed)
            mean= sess.run(iou, feed_dict=feed)
            mean_ious.append(mean)

            ground_truth_image = (gt_image[:,:,1] == True)
            gt_label = ground_truth_image.reshape(image_shape[0], image_shape[1], 1)
            gt_image_mask = np.dot(gt_label, np.array([[255, 0, 255, 127]]))
            gt_image_mask = scipy.misc.toimage(gt_image_mask, mode="RGBA")
            predicted_labels_mask = np.dot(predicted_labels, np.array([[0, 255, 0, 127]]))
            predicted_labels_mask = scipy.misc.toimage(predicted_labels_mask, mode="RGBA")

            street_im = scipy.misc.toimage(image)
            street_im.paste(gt_image_mask, box=None, mask=gt_image_mask)
            street_im.paste(predicted_labels_mask, box=None, mask=predicted_labels_mask)

            scipy.misc.imsave(os.path.join(out_dir, 'image_' + str(img_counter) + '.png'), street_im)
            img_counter += 1

        print("Evaluation Batch {0}, mean IoU: {1:.3} ".format( counter, sum(mean_ious)/len(mean_ious) ))
        counter += 1

    total_mean_iou = sum(mean_ious)/len(mean_ious)
    print("Mean IoU: ", total_mean_iou)

    return total_mean_iou

def run(train_evaluate, evaluate, model_path=''):
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    if tf.gfile.Exists(log_dir):
        tf.gfile.DeleteRecursively(log_dir)
    tf.gfile.MakeDirs(log_dir)

    learning_rate = tf.placeholder(dtype=tf.float32, name="learning_rate")
    ground_truth = tf.placeholder(dtype=tf.float32, shape=[None, image_shape[0], image_shape[1], num_classes], name="ground_truth")
    prediction = tf.placeholder(dtype=tf.float32, shape=[None, image_shape[0], image_shape[1], num_classes], name="prediction")
    iou, iou_op = mean_iou(ground_truth, prediction, num_classes)

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, l3, l4, l7 = load_vgg(sess, vgg_path)

        # Create the Decoder part
        output = layers(l3, l4, l7, num_classes)

        # Create the loss function and the optimizer
        logits, train_op, cross_entropy_loss = optimize(output, ground_truth, learning_rate, num_classes)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(log_dir)
        writer.add_graph(sess.graph)

        saver = tf.train.Saver()

        if model_path != '':
            print('Restoring model from %s' % model_path)
            saver.restore(sess, model_path)

        if not train_evaluate and not evaluate:
            # Trains the model with all images in the training folder
            print("Train the model...")
            # Create function to get batches
            gen_train_batch_fn, _ = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

            # Train the neural network
            train_nn(sess,
                     Epochs,
                     Batch_Size,
                     gen_train_batch_fn,
                     train_op,
                     cross_entropy_loss,
                     input_image,
                     ground_truth,
                     keep_prob,
                     learning_rate)

            save_path = saver.save(sess, os.path.join(models_dir, "model.ckpt"))
            print("Model saved in file: %s" % save_path)

            # Run the model with the test images and save each painted output image (roads painted green)
            helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
            # OPTIONAL: Apply the trained model to a video
        elif train_evaluate:

            # Create function to get batches for training as well as for evaluation
            gen_train_batch_fn, gen_test_batch_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'),
                                                              image_shape,
                                                              use_all_training_data=False)

            print("Train the model...")
            train_nn(sess,
                     Epochs,
                     Batch_Size,
                     gen_train_batch_fn,
                     train_op,
                     cross_entropy_loss,
                     input_image,
                     ground_truth,
                     keep_prob,
                     learning_rate,
                     summary,
                     writer)

            save_path = saver.save(sess, os.path.join(models_dir, "model.ckpt"))
            print("Model saved in file: %s" % save_path)

            print("Evaluating the model...")
            evaluate_nn(sess,
                        Batch_Size,
                        gen_test_batch_fn,
                        keep_prob,
                        logits,
                        input_image,
                        prediction,
                        ground_truth,
                        iou,
                        iou_op,
                        summary,
                        writer)
        else:
            _, gen_test_batch_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'),
                                                             image_shape,
                                                             use_all_training_data=False)
            print("Evaluating the model...")
            evaluate_nn(sess,
                        Batch_Size,
                        gen_test_batch_fn,
                        keep_prob,
                        logits,
                        input_image,
                        prediction,
                        ground_truth,
                        iou,
                        iou_op,
                        summary,
                        writer)


    print ("... Finished")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Semantic Segmentation of road pixels in images.'
                                                 'Calling main.py without any cmdline args will train a '
                                                 'model with all images available in the training set. If you call main.py -e then '
                                                 'the model will be trained with 90% of the training images and the remaining 10% '
                                                 'will be used for model evaluation')
    parser.add_argument('-t', '--train_evaluate', help='Trains and evaluates a model', action="store_true")
    parser.add_argument('-e', '--evaluate', help='Evaluates a model.', action="store_true")
    parser.add_argument('-m', '--model', type=str, help='Path to the model that shall be restored and used.', default='', nargs='?')

    args = parser.parse_args()

    print(args)

    run(args.train_evaluate, args.evaluate, args.model)
