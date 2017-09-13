import tensorflow as tf
import numpy as np
import sys
import pandas as pd
from tensorflow.contrib import learn
from azureml.logging import get_azureml_logger
from azureml.api.schema.dataTypes import DataTypes
from azureml.api.schema.sampleDefinition import SampleDefinition
import azureml.api.realtime.services as amlo16n

def evaluateModel(pred, y):
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy

def init():
    global sess, pred_op, x, phase_train, keep_prob, graph, fldr
    import_path = fldr + 'outputs/mnist/'
    print("importing model from ", import_path)
    sess = tf.Session()

    saver = tf.train.import_meta_graph(str('%s/mnistmodel.meta'%import_path))
    saver.restore(sess, tf.train.latest_checkpoint(import_path))
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    phase_train = graph.get_tensor_by_name("phase_train:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")
    pred_op = graph.get_tensor_by_name("output/final_op:0")

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

    return probs_and_class_category_df

def predict_all_test_data():
    global mnist, scores, retscores, retdf, btch_sz
    
    scores = []
    lbls = []

    for i in range(mnist.test.images.shape[0]//btch_sz):
        st = (i*btch_sz)
        et = (st+btch_sz)
        retdf = run(mnist.test.images[st:et])
        currscores = retdf['scored probabilities']
        currlbls = retdf['scored labels']
        scores = np.append(scores, retscores, axis=0) if not i==0 else retscores
        lbls = np.append(lbls, currlbls, axis=0) if not i==0 else currlbls


def main():
    global mnist, scores, sess, x, graph, btch_sz
    mnist = learn.datasets.mnist.read_data_sets('MNIST_data', one_hot=True)
    btch_sz = 5
    init()
    predict_all_test_data()
    y = graph.get_tensor_by_name("y:0")
    eval_op = evaluateModel(scores, y)
    test_feed_dict = {
        x: mnist.test.images,
        y: mnist.test.labels,


    }
    print(scores.shape)
    _netacc = sess.run(eval_op, feed_dict=test_feed_dict)
    print("Net Accuracy:", _netacc)
    print(scores[0:5,:], " predicted value = ", np.argmax(scores[0:5,:], axis=1), 
    " actual value", np.argmax(mnist.test.labels[0:5,:], axis=1))
    run_logger = get_azureml_logger() 
    run_logger.log("Accuracy",str(_netacc))
    
    print("Calling prepare schema")
    inputs = {"nparr": SampleDefinition(DataTypes.NUMPY, mnist.test.images[0:btch_sz])}
    outputs = {"probs_and_class_category_df": SampleDefinition(DataTypes.PANDAS, retdf)}
    amlo16n.generate_schema(inputs=inputs,
                            outputs=outputs,
                            filepath="outputs/mnistschema.json",
                            run_func=run
                            )

    amlo16n.generate_main(user_file="mnistscore.py", schema_file="outputs/schema.json",
                          main_file_name="outputs/main.py")
    print("End of prepare schema")

global fldr
fldr=""
if __name__ == "__main__":
    fldr = os.environ['AZUREML_NATIVE_SHARE_DIRECTORY'] 
    main()