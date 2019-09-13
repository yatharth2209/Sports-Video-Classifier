import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import os
from sklearn.metrics import confusion_matrix
import inception

import prettytensor as pt
import knifey


from knifey import num_classes

data_dir = knifey.data_dir

dataset = knifey.load()

class_names = dataset.class_names
class_names

image_paths_train, cls_train, labels_train = dataset.get_training_set()

print(image_paths_train[50])


image_paths_test, cls_test, labels_test = dataset.get_test_set()

image_paths_pred, cls_pred, labels_pred = dataset.get_pred_set()

print(image_paths_test[50])

print("Size of:")
print("- Training-set:\t\t{}".format(len(image_paths_train)))
print("- Test-set:\t\t{}".format(len(image_paths_test)))


def plot_images(images, cls_true, cls_pred=None, smooth=True):

    assert len(images) == len(cls_true)

    fig, axes = plt.subplots(3, 3)

    if cls_pred is None:
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    if smooth:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'

    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i],
                      interpolation=interpolation)

            cls_true_name = class_names[cls_true[i]]

            if cls_pred is None:
                xlabel = "True: {0}".format(cls_true_name)
            else:
                cls_pred_name = class_names[cls_pred[i]]

                xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)

            ax.set_xlabel(xlabel)
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.show()


from matplotlib.image import imread

def load_images(image_paths):
    images = [imread(path) for path in image_paths]

    return np.asarray(images)


images = load_images(image_paths=image_paths_test[0:9])

images1 = load_images(image_paths=image_paths_pred[0:9])

cls_true = cls_test[0:9]

plot_images(images=images, cls_true=cls_true, smooth=True)

inception.maybe_download()

model = inception.Inception()

from inception import transfer_values_cache

file_path_cache_train = os.path.join(data_dir, 'cnn-train.pkl')
file_path_cache_test = os.path.join(data_dir, 'cnn-test.pkl')
#file_path_cache_pred = os.path.join(data_dir, 'cnn-pred.pkl')

print("Processing Inception transfer-values for training-images ...")

transfer_values_train = transfer_values_cache(cache_path=file_path_cache_train,
                                              image_paths=image_paths_train,
                                              model=model)
print("Processing Inception transfer-values for test-images ...")

transfer_values_test = transfer_values_cache(cache_path=file_path_cache_test,
                                             image_paths=image_paths_test,
                                             model=model)

print("Processing Inception transfer-values for pred-images ...")


print(transfer_values_train.shape)

print(transfer_values_test.shape)

# print(transfer_values_pred.shape)

def plot_transfer_values(i):
    print("Input image:")
    
    image = imread(image_paths_test[i])
    plt.imshow(image, interpolation='spline16')
    plt.show()
    
    print("Transfer-values for the image using Inception model:")
    
    img = transfer_values_test[i]
    img = img.reshape((32, 64))

    plt.imshow(img, interpolation='nearest', cmap='Reds')
    plt.show()


plot_transfer_values(i=100)


plot_transfer_values(i=300)


from sklearn.decomposition import PCA

pca = PCA(n_components=2)

transfer_values = transfer_values_train

cls = cls_train

transfer_values.shape

transfer_values_reduced = pca.fit_transform(transfer_values)

transfer_values_reduced.shape

def plot_scatter(values, cls):
    import matplotlib.cm as cm
    cmap = cm.rainbow(np.linspace(0.0, 1.0, num_classes))

    idx = np.random.permutation(len(values))
    
    colors = cmap[cls[idx]]

    x = values[idx, 0]
    y = values[idx, 1]

    plt.scatter(x, y, color=colors, alpha=0.5)
    plt.show()

plot_scatter(transfer_values_reduced, cls=cls)

from sklearn.manifold import TSNE
pca = PCA(n_components=50)
transfer_values_50d = pca.fit_transform(transfer_values)
tsne = TSNE(n_components=2)

transfer_values_reduced = tsne.fit_transform(transfer_values_50d) 

transfer_values_reduced.shape


plot_scatter(transfer_values_reduced, cls=cls)

transfer_len = model.transfer_len

x = tf.placeholder(tf.float32, shape=[None, transfer_len], name='x')

y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')

y_true_cls = tf.argmax(y_true, dimension=1)

x_pretty = pt.wrap(x)

with pt.defaults_scope(activation_fn=tf.nn.relu):
    y_pred, loss = x_pretty.        fully_connected(size=1024, name='layer_fc1').        softmax_classifier(num_classes=num_classes, labels=y_true)


global_step = tf.Variable(initial_value=0,
                          name='global_step', trainable=False)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, global_step)


y_pred_cls = tf.argmax(y_pred, dimension=1,name='y_pred_cls')

correct_prediction = tf.equal(y_pred_cls, y_true_cls)


accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


saver=tf.train.Saver()
savepath="checkpoint/transfer/inception_cnn"

session = tf.Session()


session.run(tf.global_variables_initializer())

train_batch_size = 64

def random_batch():
    num_images = len(transfer_values_train)

    idx = np.random.choice(num_images,
                           size=train_batch_size,
                           replace=False)

    x_batch = transfer_values_train[idx]
    y_batch = labels_train[idx]

    return x_batch, y_batch


def optimize(num_iterations):
    start_time = time.time()

    for i in range(num_iterations):
        x_batch, y_true_batch = random_batch()

        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        i_global, _ = session.run([global_step, optimizer],
                                  feed_dict=feed_dict_train)

        if (i_global % 100 == 0) or (i == num_iterations - 1):
            batch_acc = session.run(accuracy,
                                    feed_dict=feed_dict_train)

            
            #saver.save(sess=session, save_path=savepath)
            msg = "Global Step: {0:>6}, Training Batch Accuracy: {1:>6.1%}"
            print(msg.format(i_global, batch_acc))

    end_time = time.time()

    time_dif = end_time - start_time

    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


def plot_example_errors(cls_pred, correct):
    incorrect = (correct == False)

    idx = np.flatnonzero(incorrect)

    n = min(len(idx), 9)
    
    idx = np.random.choice(idx,
                           size=n,
                           replace=False)

    cls_pred = cls_pred[idx]

    cls_true = cls_test[idx]

    image_paths = [image_paths_test[i] for i in idx]
    images = load_images(image_paths)

    plot_images(images=images,
                cls_true=cls_true,
                cls_pred=cls_pred)


from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cls_pred):
    cm = confusion_matrix(y_true=cls_test,  
                          y_pred=cls_pred)  

    for i in range(num_classes):
        class_name = "({}) {}".format(i, class_names[i])
        print(cm[i, :], class_name)

    class_numbers = [" ({0})".format(i) for i in range(num_classes)]
    print("".join(class_numbers))

    plt.matshow(cm)

    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.show()


batch_size = 256

def predict_cls(transfer_values, labels, cls_true):
    num_images = len(transfer_values)

    cls_pred = np.zeros(shape=num_images, dtype=np.int)

    i = 0

    while i < num_images:
        j = min(i + batch_size, num_images)

        feed_dict = {x: transfer_values[i:j],
                     y_true: labels[i:j]}

        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        i = j
        
    correct = (cls_true == cls_pred)

    return correct, cls_pred


def predict_cls_test():
    return predict_cls(transfer_values = transfer_values_test,
                       labels = labels_test,
                       cls_true = cls_test)


def classification_accuracy(correct):
    return correct.mean(), correct.sum()


def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):

    correct, cls_pred = predict_cls_test()
    
    acc, num_correct = classification_accuracy(correct)
    
    num_images = len(correct)

    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, num_correct, num_images))

    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)


print_test_accuracy(show_example_errors=False,
                    show_confusion_matrix=True)

optimize(num_iterations=5000)

print_test_accuracy(show_example_errors=True,
                    show_confusion_matrix=True)

saver.save(sess=session, save_path=savepath)