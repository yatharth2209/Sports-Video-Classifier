import cv2
import tensorflow as tf
import numpy as np
import glob
from random import shuffle
import os
import sys

labels=list()
addrs=list()
shuffle_data = True

print(os.getcwd())
test_path="Data_dir/*/*.jpg"

addrs = glob.glob(test_path)
for addr in addrs:
    if "baseball" in addr:
        labels.append(0)
    elif "golf" in addr:
        labels.append(1)
    elif "soccer" in addr:
        labels.append(2)
    elif "skiing" in addr:
        labels.append(3)
    elif "rowing" in addr:
        labels.append(4)
    elif "bmx" in addr:
        labels.append(5)
    elif "hammerthrow" in addr:
        labels.append(6)
    elif "tennis" in addr:
        labels.append(7)
    elif "polevault" in addr:
        labels.append(8)
    elif "hurdling" in addr:
        labels.append(9)

if shuffle_data:
    c = list(zip(addrs, labels))
    shuffle(c)
    addrs, labels = zip(*c)

train_addrs = addrs[0:int(0.7*len(addrs))]
train_labels = labels[0:int(0.7*len(labels))]
print(len(train_labels))
test_addrs = addrs[int(0.7*len(addrs)):int(0.9*len(addrs))]
test_labels = labels[int(0.7*len(labels)):int(0.9*len(addrs))]
print(len(test_labels))
valid_addrs = addrs[int(0.9*len(addrs)):]
valid_labels = labels[int(0.9*len(labels)):]
print(len(valid_labels))

def load_image(addr):
    img = cv2.imread(addr)
    img = cv2.resize(img, (100, 100), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    return img

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

train_filename = 'train.tfrecords'
writer = tf.python_io.TFRecordWriter(train_filename)
for i in range(len(train_addrs)):
    if not i % 100:
        print ('Train data: {}/{}'.format(i, len(train_addrs)))
        sys.stdout.flush()
    # Load the image
    img = load_image(train_addrs[i])
    label = train_labels[i]
    # Create a feature
    feature = {'train/label': _int64_feature(label),
               'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    
    # Serialize to string and write on the file
    writer.write(example.SerializeToString())
    
writer.close()
sys.stdout.flush()


test_filename = 'test.tfrecords'
writer = tf.python_io.TFRecordWriter(test_filename)
for i in range(len(test_addrs)):
    if not i % 100:
        print ('Test data: {}/{}'.format(i, len(test_addrs)))
        sys.stdout.flush()

    img = load_image(test_addrs[i])
    label = test_labels[i]
    feature = {'test/label': _int64_feature(label),
               'test/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())
writer.close()
sys.stdout.flush()

valid_filename = 'validation.tfrecords'
writer = tf.python_io.TFRecordWriter(valid_filename)
for i in range(len(valid_addrs)):
    if not i % 100:
        print ('Validation data: {}/{}'.format(i, len(valid_addrs)))
        sys.stdout.flush()

    img = load_image(valid_addrs[i])
    label = valid_labels[i]
    feature = {'valid/label': _int64_feature(label),
               'valid/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())
writer.close()
sys.stdout.flush()