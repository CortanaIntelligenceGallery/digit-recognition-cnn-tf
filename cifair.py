import tensorflow as tf
import numpy as np
import shutil
import os
import sys
from tensorflow.contrib import learn
from azureml.logging import get_azureml_logger

def splayer(inp, wt_shp, b_shp, isFinal=False, batch_norm =False, phase_train=False):
    wt_std = (2.0/wt_shp[0])**0.5
    w_i = tf.random_normal_initializer(stddev=wt_std)
    b_i = tf.constant_initializer(value=0)
    W = tf.get_variable("W", wt_shp, initializer=w_i)
    b = tf.get_variable("b", b_shp, initializer=b_i)
    logits = tf.matmul(inp, W)+b
    norm_logits = logits if not batch_norm else layer_batch_norm(logits, wt_shp[1], phase_train=phase_train)

    return tf.nn.relu(norm_logits) if not isFinal else tf.nn.softmax(norm_logits, name='final_op')

def sploss(output, y):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y)
    loss = tf.reduce_mean(xentropy)
    tf.summary.scalar('loss', loss)
    return loss

def conv2d(inp, wt_shp, bias_shp, batch_norm=False, phase_train=False):
    in1 = wt_shp[0] * wt_shp[1] * wt_shp[2]

    wt_init = tf.random_normal_initializer(stddev=(2.0/in1)**0.5)
    W = tf.get_variable("W", wt_shp, initializer=wt_init)
    bias_init = tf.constant_initializer(value=0)
    b = tf.get_variable("b", bias_shp, initializer=bias_init)
    conv_out = tf.nn.conv2d(inp, W, strides=[1,1,1,1],
                            padding='SAME')
    logits = tf.nn.bias_add(conv_out, b)

    return tf.nn.relu(logits) if not batch_norm else tf.nn.relu(conv_batch_norm(logits, wt_shp[3], phase_train=phase_train))

def max_pool(inp, k=2):
    return tf.nn.max_pool(inp, ksize=[1,k,k,1],
                          strides=[1,k,k,1], padding='SAME')

def inference(x, keep_prob, batch_norm=False, phase_train=False):
    x = tf.reshape(x, shape=[-1,28,28,1])
    with tf.variable_scope("conv_1"):
        conv_1 = conv2d(x, [5,5,1,32],[32], batch_norm=batch_norm, phase_train=phase_train)
        pool_1 = max_pool(conv_1)
    with tf.variable_scope("conv_2"):
        conv_2 = conv2d(pool_1, [5,5,32,64], [64], batch_norm=batch_norm, phase_train=phase_train)
        pool_2 = max_pool(conv_2)
    with tf.variable_scope("fc"):
        pool_2_flat = tf.reshape(pool_2, [-1, 7*7*64])
        fc_1 = splayer(pool_2_flat, [7*7*64, 1024], [1024], isFinal=False, batch_norm=batch_norm, phase_train=phase_train)
        fc_l_drop = tf.nn.dropout(fc_1, keep_prob)
    with tf.variable_scope("output"):
        output = splayer(fc_l_drop, [1024, 10], [10], isFinal=True, batch_norm=batch_norm, phase_train=phase_train)
    return output

def conv_batch_norm(x, n_out, phase_train):
    beta_init = tf.constant_initializer(value=0.0, dtype=tf.float32)
    gamma_init = tf.constant_initializer(value=1.0,
                                         dtype=tf.float32)
    beta = tf.get_variable("beta", [n_out], initializer=beta_init)
    gamma = tf.get_variable("gamma", [n_out], initializer=gamma_init)
    batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.9)
    ema_apply_op = ema.apply([batch_mean, batch_var])
    ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)

    def mean_var_with_update():
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(phase_train, mean_var_with_update,lambda: (ema_mean, ema_var))
    normed = tf.nn.batch_norm_with_global_normalization(x, mean, var, beta, gamma, 1e-3, True)
    return normed

def layer_batch_norm(x, n_out, phase_train):
    beta_init = tf.constant_initializer(value=0.0, dtype=tf.float32)
    gamma_init = tf.constant_initializer(value=1.0,
                                         dtype=tf.float32)
    beta = tf.get_variable("beta", [n_out], initializer=beta_init)
    gamma = tf.get_variable("gamma", [n_out], initializer=gamma_init)
    batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.9)
    ema_apply_op = ema.apply([batch_mean, batch_var])
    ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)

    def mean_var_with_update():
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)


    mean, var = tf.cond(phase_train, mean_var_with_update,lambda: (ema_mean, ema_var))
    normed = tf.nn.batch_norm_with_global_normalization(x, mean, var, beta, gamma, 1e-3, True)
    return tf.reshape(normed, [-1, n_out])

def downloaddata():
    global mnist    
    mnist = learn.datasets.mnist.read_data_sets('MNIST_data', one_hot=True)

def evaluateModel(pred, y):
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    return accuracy

def train(loss):
    train_step = tf.train.AdamOptimizer().minimize(loss)
    return train_step

def sessionrun(num_epochs):
    global mnist, serialized_tf_example, prediction_classes, values
    global tensor_info_x, tensor_info_y, sessinfo
    global train_x, train_y
    downloaddata()
    
    batch_size = 100

    x = tf.placeholder(tf.float32, [None, 784], name='x')
    y = tf.placeholder(tf.float32, [None, 10], name='y')
    phase_train = tf.placeholder(tf.bool, name='phase_train')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    batch_norm = tf.placeholder(tf.bool, name='batch_norm')

    pred_op = inference(x, keep_prob, batch_norm=True, phase_train=phase_train)
    loss_op = sploss(pred_op, y)
    ts_op = train(loss_op)
    eval_op = evaluateModel(pred_op, y)
    values, indices = tf.nn.top_k(pred_op, 10)

    loss_list = []
    acc_list = []
    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter('outputs/tflogs/train', sess.graph)
        test_writer = tf.summary.FileWriter('outputs/tflogs/test')

        sess.run(tf.global_variables_initializer())
        saver0 = tf.train.Saver()
        for epoch in range(num_epochs):
            avgloss = 0.
            avgacc = 0.

            total_batch = int(mnist.train.num_examples/batch_size)
            for i in range(total_batch):
                mx, my = mnist.train.next_batch(batch_size)
                #nx = 1-mx - this is for training images on whitebackground

                feed_dict = {x: mx, y:my, phase_train:True, batch_norm: True, keep_prob:0.4}
                _trsumm, _totloss, _trainstep, _predseriescc = sess.run(
                    [merged, loss_op, ts_op, pred_op],
                    feed_dict=feed_dict)
                avgloss += _totloss/total_batch
                #this is for training images on whitebackground
                #feed_dict = {x: nx, y: my, phase_train: True, batch_norm: True, keep_prob: 0.4}
                #_totloss, _trainstep, _predseriescc = sess.run(
                #    [loss_op, ts_op, pred_op],
                #    feed_dict=feed_dict)
                #avgloss += _totloss / total_batch
                loss_list.append(avgloss)
                if (i%10==0):
                    train_writer.add_summary(_trsumm, i)
            val_feed_dict = {
                x: mnist.validation.images,
                y: mnist.validation.labels,
                phase_train: False,
                batch_norm: True,
                keep_prob: 1
            }
            _valsumm,_acc = sess.run([merged, eval_op], feed_dict=val_feed_dict)
            avgacc = _acc
            acc_list.append(_acc)
            print("In Epoch ", epoch, " with loss ", avgloss, " and with accuracy ", avgacc)
            train_writer.add_summary(_trsumm, epoch*batch_size)
            test_writer.add_summary(_valsumm, epoch)

        print("validating test data")
        test_feed_dict = {
            x: mnist.test.images,
            y: mnist.test.labels,
            phase_train: False,
            batch_norm: True,
            keep_prob: 1
        }

        print("starting test run")
        _tstsumm,_netacc = sess.run([merged, eval_op], feed_dict= test_feed_dict)
        print("test run completed")
        sys.stdout.flush()
        tensor_info_x = tf.saved_model.utils.build_tensor_info(x)
        tensor_info_y = tf.saved_model.utils.build_tensor_info(pred_op)
        # initialize the logger
        print("Net accuracy: ", _netacc)
        run_logger = get_azureml_logger()   
        run_logger.log("Accuracy ", str(_netacc))
        print("Number of epochs: ", num_epochs)
        run_logger.log("Number of Epochs", str(num_epochs))

        

        # export model to outputs folder
        print("export model to outputs folder")
        export_path_base = 'outputs/mnist/'
        print('export_path_base:', export_path_base)
        if os.path.exists(export_path_base):
           print("model path already exist, removing model path files and directory")
           shutil.rmtree(export_path_base)
        os.makedirs(export_path_base, exist_ok=True)
        saver0.save(sess, export_path_base+'mnistmodel')
        print('Done exporting!')

        print("export model to azure share folder")
        export_path_base = os.environ['AZUREML_NATIVE_SHARE_DIRECTORY'] + 'outputs/mnist/'
        print('export_path_base:', export_path_base)
        if os.path.exists(export_path_base):
           print("model path already exist, removing model path files and directory")
           shutil.rmtree(export_path_base)
        os.makedirs(export_path_base, exist_ok=True)
        saver0.save(sess, export_path_base+'mnistmodel')
        print('Done exporting!')


def main():
    
    sessionrun(5)

main()


