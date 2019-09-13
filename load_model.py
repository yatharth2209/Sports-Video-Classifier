import tensorflow as tf
import numpy as np
from read_tfrecord import predict_me

prediction_images = predict_me()
literal_labels=["baseball", "golf", "soccer", "skiing", "rowing", "bmx","hammerthrow","tennis","polevault","hurdling"]

with tf.Session() as session:
    
    def new_prediction():
        saver = tf.train.import_meta_graph('checkpoint/cnn.meta')
        saver.restore(session,tf.train.latest_checkpoint('checkpoint/'))
        graph=tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        #transfer_values_pred=graph.get_tensor_by_name("transfer_values_pred:0")
        #y_true = graph.get_tensor_by_name("y_true:0")
        y_pred_cls=graph.get_tensor_by_name("y_pred_cls:0")
        feed_dict_predict={x:prediction_images}
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
        if (max(probabilities) < 0.4):
        	print("\nCannot understand the unique class of the given video.")
        else:
        	print("Input video is of people playing " +unique_labels[ind])
    new_prediction()