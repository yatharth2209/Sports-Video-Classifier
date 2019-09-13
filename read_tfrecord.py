import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

num_classes=10


def train_me():
    data_path = 'train.tfrecords'
    with tf.Session() as sess:
        feature = {'train/image': tf.FixedLenFeature([], tf.string),
                   'train/label': tf.FixedLenFeature([], tf.int64)}
        filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example, features=feature)
        image = tf.decode_raw(features['train/image'], tf.float32)
        label = tf.cast(features['train/label'], tf.int32)
        image = tf.reshape(image, [100, 100, 3])
        images, labels = tf.train.shuffle_batch([image, label], batch_size=100, capacity=30, num_threads=2, min_after_dequeue=10)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for batch_index in range(num_classes+1):
            img, lbl = sess.run([images, labels])
            img = img.astype(np.uint8)
            one_hot_lbl=np.eye(num_classes)[lbl]
        coord.request_stop()
        coord.join(threads)
        sess.close()
        img=np.reshape(img, (len(img), 10000, 3))
        return img, one_hot_lbl

def try_me():
    train_batch_size=3500
    data_path = 'train.tfrecords'
    with tf.Session() as sess:
        feature = {'train/image': tf.FixedLenFeature([], tf.string),
                   'train/label': tf.FixedLenFeature([], tf.int64)}
        filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example, features=feature)
        image = tf.decode_raw(features['train/image'], tf.float32)
        
        label = tf.cast(features['train/label'], tf.int32)
        image = tf.reshape(image, [100, 100, 3])
        
        images, labels = tf.train.shuffle_batch([image, label], batch_size=train_batch_size, capacity=30, num_threads=2, min_after_dequeue=10)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        img, lbl = sess.run([images, labels])
        img = img.astype(np.uint8)
        one_hot_lbl=np.eye(num_classes)[lbl]
        coord.request_stop()
        coord.join(threads)
        sess.close()
        img=np.reshape(img, (len(img), 10000, 3))
        return img, one_hot_lbl, lbl


def test_me():
    test_batch_size=1000
    data_path = 'test.tfrecords'
    with tf.Session() as sess:
        feature = {'test/image': tf.FixedLenFeature([], tf.string),
                   'test/label': tf.FixedLenFeature([], tf.int64)}
        filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example, features=feature)
        image = tf.decode_raw(features['test/image'], tf.float32)
        
        label = tf.cast(features['test/label'], tf.int32)
        image = tf.reshape(image, [100, 100, 3])
        #image = tf.image.rgb_to_grayscale(image)
        
        images, labels = tf.train.shuffle_batch([image, label], batch_size=test_batch_size, capacity=30, num_threads=2, min_after_dequeue=10)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        img, lbl = sess.run([images, labels])
        img = img.astype(np.uint8)
        one_hot_lbl=np.eye(num_classes)[lbl]
        coord.request_stop()
        
        coord.join(threads)
        sess.close()
        #print(len(img))
        img=np.reshape(img, (len(img), 10000, 3))
        return img, one_hot_lbl, lbl


def validate_me():
    valid_batch_size=500
    data_path = 'validation.tfrecords'
    with tf.Session() as sess:
        feature = {'valid/image': tf.FixedLenFeature([], tf.string),
                   'valid/label': tf.FixedLenFeature([], tf.int64)}
        filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example, features=feature)
        image = tf.decode_raw(features['valid/image'], tf.float32)
        
        label = tf.cast(features['valid/label'], tf.int32)
        image = tf.reshape(image, [100, 100, 3])
        #image = tf.image.rgb_to_grayscale(image)
        
        images, labels = tf.train.shuffle_batch([image, label], batch_size=valid_batch_size, capacity=30, num_threads=2, min_after_dequeue=10)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        img, lbl = sess.run([images, labels])
        img = img.astype(np.uint8)
        one_hot_lbl=np.eye(num_classes)[lbl]
        coord.request_stop()
        
        coord.join(threads)
        sess.close()
        #print(len(img))
        img=np.reshape(img, (len(img), 10000, 3))
        return img, one_hot_lbl, lbl

def predict_me():
    data_path = 'predict.tfrecords'
    with open('predict_info.txt') as f:
        content = f.read().splitlines()
    for line in content:
        if line.startswith("Predict"):
            b_size=int(line.split()[1])
    with tf.Session() as sess:
        feature = {'predict/image': tf.FixedLenFeature([], tf.string),
                   'predict/label': tf.FixedLenFeature([], tf.int64)}
        filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example, features=feature)
        image = tf.decode_raw(features['predict/image'], tf.float32)
        
        label = tf.cast(features['predict/label'], tf.int32)
        image = tf.reshape(image, [100, 100, 3])
        #image = tf.image.rgb_to_grayscale(image)
        images, labels = tf.train.shuffle_batch([image, label], batch_size=b_size, capacity=30, num_threads=2, min_after_dequeue=10)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        img, lbl = sess.run([images, labels])
        img = img.astype(np.uint8)
        coord.request_stop()
        coord.join(threads)
        sess.close()
        img=np.reshape(img, (len(img), 10000, 3))
        return img

d1=predict_me()
# print(d1[0])
# # print(len(d1[0]))
# # print(len(d1[0][0]))
# # print(len(d1[0][0][0]))

# print(len(d1))

# print(len(x))

