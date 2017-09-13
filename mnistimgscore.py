import tensorflow as tf
import numpy as np
import sys
import pandas as pd
from tensorflow.contrib import learn
from azureml.api.schema.dataTypes import DataTypes
from azureml.api.schema.sampleDefinition import SampleDefinition
import azureml.api.realtime.services as amlo16n
from PIL import Image

def load_image(infilename):
    img = Image.open(infilename)
    img.load()
    data = np.asarray(img, dtype="int32" )
    return data

def conv_to_single_dim(arr, normalized=True):
    newarr = np.zeros([arr.shape[0], arr.shape[1]])

    newarr = (arr[:, :, 0] + arr[:, :, 1] + arr[:, :, 2]) / 3
    newarr = np.reshape(newarr, [784])
    for i in range(len(newarr)):
        if (normalized):
            newarr[i] = 1 - (newarr[i] / 255)
            newarr[i] = newarr[i] if newarr[i] > 0.5 else 0
        else:
            newarr[i] = (newarr[i] / 255)
    return newarr

def conv_str_to_img(npstr):
    img = npstr

    return img

def init():
    global sess, pred_op, x, phase_train, keep_prob, graph, fldr
    import_path = fldr+'outputs/mnist/'
    #import_path = './outputs/mnist/'

    sess = tf.Session()

    saver = tf.train.import_meta_graph(str('%s/mnistmodel.meta'%import_path))
    saver.restore(sess, tf.train.latest_checkpoint(import_path))
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    phase_train = graph.get_tensor_by_name("phase_train:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")
    pred_op = graph.get_tensor_by_name("output/final_op:0")
    writer = tf.train.write_graph(sess.graph, logdir="tflogs", name="CNN")

def run(nparr):
    global sess, pred_op, x, phase_train, keep_prob, retscores
    feed_dict = {
        x: nparr,
        keep_prob: 1.0,
        phase_train: False
    }
    retscores = sess.run(pred_op, feed_dict=feed_dict)
    retlbls = np.argmax(retscores, axis=1)
    retprobs = np.amax(retscores, axis=1)

    probs_and_class_category_df = pd.DataFrame(data={"scored probabilities":retprobs, "scored labels":retlbls})
    return str(probs_and_class_category_df)

def predict_img(fname):
    arr = load_image(fname)
    arr = conv_to_single_dim(arr, normalized=False)
    print(fname,end='')
    print(run([arr]),end='')
    print()

def main():
    global retscores
    print("main")
    init()
    
    predict_img("images/2.png")

global fldr
fldr=""
if __name__ == "__main__":
    print("calling main")
    fldr = os.environ['AZUREML_NATIVE_SHARE_DIRECTORY'] 
    main()
