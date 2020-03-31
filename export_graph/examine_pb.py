import tensorflow as tf
import sys
import os
from tensorflow.python.platform import gfile

from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat

with tf.Session() as sess:
    model_filename =os.path.join(os.getcwd(), 'frozen_inference_graph.pb')
    with gfile.FastGFile(model_filename, 'rb') as f:

        data = compat.as_bytes(f.read())
        sm = saved_model_pb2.SavedModel()
        sm.ParseFromString(data)

        if 1 != len(sm.meta_graphs):
            print('More than one graph found. Not sure which to write')
            sys.exit(1)

        #graph_def = tf.GraphDef()
        #graph_def.ParseFromString(sm.meta_graphs[0])
        g_in = tf.import_graph_def(sm.meta_graphs[0].graph_def)

    LOGDIR=os.getcwd()
    train_writer = tf.compat.v1.summary.FileWriter(LOGDIR)
    train_writer.add_graph(sess.graph)
    print("Model Imported. Visualize by running: "
          "tensorboard --logdir={}".format(LOGDIR))