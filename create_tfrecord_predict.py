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
test_path="Predict/*.jpg"

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
    elif "hurlding" in addr:
        labels.append(9)
    else:
        labels.append(10)

if shuffle_data:
    c = list(zip(addrs, labels))
    shuffle(c)
    addrs, labels = zip(*c)

print(len(addrs))
print(len(labels))

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

train_filename = 'predict.tfrecords'
writer = tf.python_io.TFRecordWriter(train_filename)
with open('predict_info.txt', 'w') as the_file:
    the_file.write("Predict "+str(len(addrs)))
for i in range(len(addrs)):
    if not i % 100 or i==len(addrs)-1:
        print ('Predicted data: {}/{}'.format(i, len(addrs)))
        sys.stdout.flush()
    img = load_image(addrs[i])
    label = labels[i]
    
    feature = {'predict/label': _int64_feature(label),
                'predict/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    
    writer.write(example.SerializeToString())
    
writer.close()
sys.stdout.flush()