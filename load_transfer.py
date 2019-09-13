import tensorflow as tf
import numpy as np
from read_tfrecord import predict_me
import inception
import os
import knifey


from knifey import num_classes

data_dir = knifey.data_dir

dataset = knifey.load()

model = inception.Inception()

from inception import transfer_values_cache

image_paths_pred, cls_pred, labels_pred = dataset.get_pred_set()

file_path_cache_pred = os.path.join(data_dir, 'cnn-pred.pkl')

transfer_values_pred = transfer_values_cache(cache_path=file_path_cache_pred,
                                             image_paths=image_paths_pred,
                                             model=model)

literal_labels=["skiing", "hurdling", "bmx", "rowing", "baseball","polevault","hammerthrow","tennis","soccer","golf"]

with tf.Session() as session:
    
    def new_prediction():
        saver = tf.train.import_meta_graph('checkpoint/transfer/inception_cnn.meta')
        saver.restore(session,tf.train.latest_checkpoint('checkpoint/transfer'))
        graph=tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        y_pred_cls=graph.get_tensor_by_name("y_pred_cls:0")
        feed_dict_predict={x:transfer_values_pred}
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
        ind=probabilities.index(max(probabilities))
        if (max(probabilities) < 0.6):
            print("\nCannot understand the unique class of the given video.")
        else:
            print("Input video is of people playing " +unique_labels[ind])

    new_prediction()