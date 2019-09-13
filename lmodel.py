import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
from read_tfrecord import train_me
from read_tfrecord import test_me
from read_tfrecord import try_me
from read_tfrecord import predict_me

img_size = 100
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)

num_classes = 6

literal_labels=["baseball", "golf", "soccer", "skiing", "rowing", "bmx"]

training_images,training_labels_one_hot, training_labels=try_me()
testing_images, testing_labels_one_hot, testing_labels=test_me()

prediction_images = predict_me()

# print(training_images[0:9])
# print(training_labels[0:9])
# # print(training_labels_one_hot[0:9])
# matplotlib.rcParams["backend"]="TkAgg"
# plt.switch_backend("TkAgg")

print("Size of:")
print("- Training-set:\t\t{}".format(len(training_labels)))
print("- Test-set:\t\t{}".format(len(testing_labels)))

x = tf.placeholder(tf.float32, [None, img_size_flat], name="x")
y_true = tf.placeholder(tf.float32, [None, num_classes],name="y_true")
y_true_cls = tf.placeholder(tf.int64, [None],name="y_true_cls")
weights = tf.Variable(tf.zeros([img_size_flat, num_classes]),name="weights")
biases = tf.Variable(tf.zeros([num_classes]),name="biases")

logits = tf.matmul(x, weights) + biases

y_pred = tf.nn.softmax(logits)
y_pred_cls = tf.argmax(y_pred, dimension=1)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                        labels=y_true)
cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session = tf.Session()

session.run(tf.global_variables_initializer())


def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(literal_labels[cls_true[i]])
        else:
            xlabel = "True: {0}, Pred: {1}".format(literal_labels[cls_true[i]], literal_labels[cls_pred[i]])

        ax.set_xlabel(xlabel)
        
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


plot_images(images=testing_images[0:9], cls_true=testing_labels[0:9])

def optimize(num_iterations):
    for i in range(num_iterations):
        x_batch, y_true_batch = train_me()
        feed_dict_train = {x: x_batch,y_true: y_true_batch}
        session.run(optimizer, feed_dict=feed_dict_train)


img, one_hot_lbl, lbl = test_me()
feed_dict_test = {x: img,  y_true: one_hot_lbl, y_true_cls: lbl}

def print_accuracy():
    acc = session.run(accuracy, feed_dict=feed_dict_test)
    print("Accuracy on test-set: {0:.1%}".format(acc))

def print_confusion_matrix():
    cls_true = lbl
    cls_pred = session.run(y_pred_cls, feed_dict=feed_dict_test)
    cm = confusion_matrix(y_true=cls_true,y_pred=cls_pred)
    print(cm)

    # Plot the confusion matrix as an image.
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    # Make various adjustments to the plot.
    plt.tight_layout()
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
    # print(type(new_pred))
    # print(new_pred)
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


def plot_example_errors():
    correct, cls_pred = session.run([correct_prediction, y_pred_cls],
                                    feed_dict=feed_dict_test)

    incorrect = (correct == False)

    images = testing_images[incorrect]
    
    cls_pred = cls_pred[incorrect]

    cls_true = testing_labels[incorrect]
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])
    plt.show()
    
def plot_weights():
    w = session.run(weights)
    
    w_min = np.min(w)
    w_max = np.max(w)

    fig, axes = plt.subplots(3, 4)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        if i<6:
            image = w[:, i].reshape(img_shape)

            ax.set_xlabel("Weights: {0}".format(i))

            ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')

        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

for i in range(300):
    iter=1
    print(str(i*iter)+" Iterations Completed")
    optimize(num_iterations=iter)
    print_accuracy()
print_confusion_matrix()
plot_example_errors()
plot_weights()
new_prediction()

# saver = tf.train.Saver()
# saver.save(session, "/home/yatharth/Desktop/Videos/linear-model")
# session.close()
