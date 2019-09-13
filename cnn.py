import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
from read_tfrecord import train_me
from read_tfrecord import test_me
from read_tfrecord import try_me
from read_tfrecord import predict_me
from read_tfrecord import validate_me

filter_size1 = 5          
num_filters1 = 16         

filter_size2 = 5          
num_filters2 = 36         
fc_size = 128             


training_images,training_labels_one_hot, training_labels=try_me()
testing_images, testing_labels_one_hot, testing_labels=test_me()
validation_images, validation_labels_one_hot, validation_labels = validate_me()

literal_labels=["baseball", "golf", "soccer", "skiing", "rowing", "bmx","hammerthrow","tennis","polevault","hurdling"]

print("Size of:")
print("- Training-set:\t\t{}".format(len(training_labels)))
print("- Test-set:\t\t{}".format(len(testing_labels)))
print("- Validation-set:\t{}".format(len(validation_labels)))

img_size = 100

img_size_flat = img_size * img_size

num_channels = 3

img_shape = (img_size, img_size, num_channels)

num_classes = 10

def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        if cls_pred is None:
            xlabel = "True: {0}".format(literal_labels[cls_true[i]])
        else:
            xlabel = "True: {0}, Pred: {1}".format(literal_labels[cls_true[i]], literal_labels[cls_pred[i]])

        ax.set_xlabel(xlabel)
        
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

plot_images(images=testing_images[0:9], cls_true=testing_labels[0:9])

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


def new_conv_layer(input,              
                   num_input_channels, 
                   filter_size,        
                   num_filters,        
                   use_pooling=True):  

    shape = [filter_size, filter_size, num_input_channels, num_filters]

    weights = new_weights(shape=shape)

    biases = new_biases(length=num_filters) #4d tesor like weights

    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    layer += biases

    if use_pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    layer = tf.nn.relu(layer)
    return layer, weights


def flatten_layer(layer):
    layer_shape = layer.get_shape()

    num_features = layer_shape[1:4].num_elements()
    
    layer_flat = tf.reshape(layer, [-1, num_features])

    return layer_flat, num_features

def new_fc_layer(input,          
                 num_inputs,     
                 num_outputs,    
                 use_relu=True): 

    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    layer = tf.matmul(input, weights) + biases

    if use_relu:
        layer = tf.nn.relu(layer)

    return layer



x = tf.placeholder(tf.float32, shape=[None, img_size_flat, num_channels], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

layer_conv1, weights_conv1 =     new_conv_layer(input=x_image,
                   num_input_channels=num_channels,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True)


layer_conv1

layer_conv2, weights_conv2 =     new_conv_layer(input=layer_conv1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   use_pooling=True)

layer_conv2

layer_flat, num_features = flatten_layer(layer_conv2)

layer_flat

num_features

layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=True)

layer_fc1

layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=num_classes,
                         use_relu=False)

layer_fc2

y_pred = tf.nn.softmax(layer_fc2)

y_pred_cls = tf.argmax(y_pred, dimension=1, name="y_pred_cls")

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_fc2,labels=y_true)

cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Save
saver=tf.train.Saver()
savepath="checkpoint/cnn"

#Create Session
session = tf.Session()


session.run(tf.global_variables_initializer())

best_validation_accuracy = 0.0
last_improvement = 0
require_improvement = 100
total_iterations = 0

def optimize(num_iterations):
    global total_iterations
    global best_validation_accuracy
    global last_improvement

    start_time = time.time()

    for i in range(num_iterations):

        total_iterations +=1
        x_batch, y_true_batch = train_me()
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        session.run(optimizer, feed_dict=feed_dict_train)

        if (total_iterations % 10 == 0) or (i == (num_iterations - 1)):
            acc_train = session.run(accuracy, feed_dict=feed_dict_train)
            acc_validation, _ = validation_accuracy()

            if acc_validation > best_validation_accuracy:
                best_validation_accuracy = acc_validation
                
                last_improvement = total_iterations

                saver.save(sess=session, save_path=savepath)

                improved_str = '*'
            else:
                improved_str = ''
            
            msg = "Iter: {0:>6}, Train-Batch Accuracy: {1:>6.1%}, Validation Acc: {2:>6.1%} {3}"

            print(msg.format(i + 1, acc_train, acc_validation, improved_str))

        if total_iterations - last_improvement > require_improvement:
            print("No improvement found in a while, stopping optimization.")
            break

    end_time = time.time()

    time_dif = end_time - start_time

    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

def plot_example_errors(cls_pred, correct):
    incorrect = (correct == False)
    
    images = testing_images[incorrect]
    
    cls_pred = cls_pred[incorrect]

    cls_true = testing_labels[incorrect]
    
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])


def plot_confusion_matrix(cls_pred):
    cls_true = testing_labels
    
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    print(cm)

    plt.matshow(cm)

    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.show()

def new_prediction():
    feed_dict_predict={x: prediction_images}
    new_pred=session.run(y_pred_cls,feed_dict=feed_dict_predict)
    unique, counts = np.unique(new_pred, return_counts=True)
    unique_labels=[]
    probabilities=[]
    for tup in counts:
        tup=int(tup)/len(new_pred)
        probabilities.append(float("{0:.2f}".format(tup)))
    for un in unique:
        un=literal_labels[int(un)]
        unique_labels.append(un)
    probability_dist=dict(zip(unique_labels, probabilities))
    print(probability_dist)


test_batch_size = 1000
batch_size = 80


def predict_cls(images, labels, cls_true):
    num_images = len(images)

    cls_pred = np.zeros(shape=num_images, dtype=np.int)

    i = 0

    while i < num_images:
        j = min(i + batch_size, num_images)

        feed_dict = {x: images[i:j, :],
                     y_true: labels[i:j, :]}

        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        i = j

    correct = (cls_true == cls_pred)

    return correct, cls_pred

def predict_cls_test():
    return predict_cls(images = testing_images,
                       labels = testing_labels_one_hot,
                       cls_true = testing_labels)

def predict_cls_validation():
    return predict_cls(images = validation_images,
                       labels = validation_labels_one_hot,
                       cls_true = validation_labels)

def cls_accuracy(correct):
    correct_sum = correct.sum()

    acc = float(correct_sum) / len(correct)

    return acc, correct_sum

def validation_accuracy():
    correct, _ = predict_cls_validation()
    return cls_accuracy(correct)



def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):

    correct, cls_pred = predict_cls_test()

    acc, num_correct = cls_accuracy(correct)
    
    num_images = len(correct)

    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, num_correct, num_images))

    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    if show_confusion_matrix:
        print("Confusion Matrix:")
        
        plot_confusion_matrix(cls_pred=cls_pred)

def plot_conv_weights(weights, input_channel=0):
    w = session.run(weights)

    w_min = np.min(w)
    w_max = np.max(w)

    num_filters = w.shape[3]

    num_grids = math.ceil(math.sqrt(num_filters))
    
    fig, axes = plt.subplots(num_grids, num_grids)

    for i, ax in enumerate(axes.flat):
        if i<num_filters:
            img = w[:, :, input_channel, i]

            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.show()


def plot_conv_layer(layer, image):
    feed_dict = {x: [image]}

    values = session.run(layer, feed_dict=feed_dict)

    num_filters = values.shape[3]

    num_grids = math.ceil(math.sqrt(num_filters))
    
    fig, axes = plt.subplots(num_grids, num_grids)

    for i, ax in enumerate(axes.flat):
        
        if i<num_filters:
            img = values[0, :, :, i]

            ax.imshow(img, interpolation='nearest', cmap='binary')
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.show()

def plot_image(image):
    plt.imshow(image.reshape(img_shape),
               interpolation='nearest',
               cmap='binary')

    plt.show()


print_test_accuracy()
plot_conv_weights(weights=weights_conv1)
optimize(num_iterations=30)
print_test_accuracy(show_example_errors=True,
                    show_confusion_matrix=True)


