
import tensorflow as tf
import os
# from tensorflow.contrib.data import Dataset, Iterator
import numpy as np
# from dataProcessing import *
import math
import GPUtil
import matplotlib.pyplot as plt
import cv2
batchSize = 30
img_height = 224
img_width = 224
num_channels = 3
patchSize = 5
learningRate = 0.00001
weight_decay = 1e-4
num_classes = 11
n_epochs = 100
model_path = 'modelNew'

trainImages = np.load('producttrainNormalImages.npy')
valImages = np.load('productvalNormalImages.npy')
testImages = np.load('producttestNormalImages.npy')
trainLabels = np.load('producttrainNormalLabels.npy')
valLabels = np.load('productvalNormalLabels.npy')
testLabels = np.load('producttestNormalLabels.npy')

VGG_MEAN = [103.939, 116.779, 123.68]
data_dict = np.load('vgg19.npy', encoding='latin1').item()


def random_crop(image, label):
    random_crop_with_mask = lambda image, label: tf.unstack(
        tf.random_crop(tf.concat((image, label), axis=-1), self._crop_shape), axis=-1)
    channel_list = random_crop_with_mask(image, label)
    image = tf.transpose(channel_list[:-1], [1,2,0])
    label = tf.expand_dims(channel_list[-1], axis=-1)
    return image, label

def random_flip(image, label):
    im_la = tf.unstack(tf.image.random_flip_left_right(tf.concat((image, label), axis=-1)), num=24,axis=-1)
    image = tf.transpose(im_la[:3], [1,2,0])
    label = tf.transpose(im_la[3:], [1,2,0])
    return (image, label)


def conv_layer(bottom, inchannels, outchannels, name, shape=[], isRelu = True):
    # print(name)
    with tf.variable_scope(name):
        # if name in data_dict:
        filt, conv_biases = get_var(name)
        # else:
        #     filt, conv_biases = get_filter_and_bias(shape)
        #     tf.summary.histogram('added_filters'+name, filt)
        
        conv = tf.nn.conv2d(bottom,filt,[1,1,1,1],padding='SAME')
        bias = tf.nn.bias_add(conv, conv_biases)
        if isRelu:
            bias = tf.nn.relu(bias)

        return bias

def fc_layer(bottom, inchannels, outchannels, name):
    with tf.variable_scope(name):
        weight, conv_biases = get_var(name)

        if name == 'fc6':
        	weight = tf.reshape(weight,[7,7,512,4096])
        else:
        	weight = tf.reshape(weight,[1,1,4096,4096])

        conv = tf.nn.conv2d(bottom,weight,[1,1,1,1],padding='SAME')
        bias = tf.nn.bias_add(conv, conv_biases)

        return bias

def get_filter_and_bias(name, shape):
    # init = tf.constant_initializer(value=tf.truncated_normal(shape))
    initializer = tf.contrib.layers.xavier_initializer()
    weights = tf.get_variable(shape=shape,name=name+'weights',initializer=initializer)
    bias = tf.Variable(tf.constant(0.0,shape=[shape[3]]),name=name+'biases')
    return weights, bias


def get_var(name):
    # print(name)
    # print(data_dict[name]['weights'])
    # if name in data_dict:
    init = tf.constant_initializer(value=data_dict[name][0],dtype=tf.float32)
    # else:
    #     init = tf.constant_initializer(value=tf.truncated_normal)
    shape = data_dict[name][0].shape
    filt = tf.get_variable(name="filter", initializer=init, shape=shape)
    # filt = tf.Variable(data_dict[name][0],name=name+'filt')

    init = tf.constant_initializer(data_dict[name][1],
                                       dtype=tf.float32)
    shape = data_dict[name][1].shape
    bias = tf.get_variable(name="biases", initializer=init, shape=shape)
    # bias = tf.Variable(data_dict[name][1],name=name+'bias')

    return filt, bias

def get_variable_with_decay(shape,stddev, name):
    weights = tf.Variable(tf.truncated_normal(shape=shape, stddev = stddev), name=name)
    # weights_with_decay = tf.multiply(tf.nn.l2_loss(weights), weight_decay)
    # # print(weights)
    # tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weights_with_decay)

    return weights

def get_bias_variable(shape,name, constant=0.0):
    return tf.Variable(tf.constant(constant,shape=shape),name=name)

def score_layer(bottom, num_classes, name):
    with tf.variable_scope(name) :
    	# print(bottom)
        infeatures = bottom.get_shape()[3].value
        shape = [1,1,infeatures,num_classes]

        stddev = (2/infeatures)**0.5
        weights = get_variable_with_decay(shape,stddev, name)
        # print(weights)

        biases = get_bias_variable([num_classes],name, 0.0)

        conv = tf.nn.conv2d(bottom,weights,[1,1,1,1], padding='SAME')
        bias_add = tf.nn.bias_add(conv,biases)

        return bias_add 

def get_deconv_filter(shape):
    width = shape[0]
    height = shape[1]
    f = math.ceil(width/2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([shape[0], shape[1]])
    for x in range(width):
        for y in range(height):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weights = np.zeros(shape)
    for i in range(shape[2]):
        weights[:, :, i, i] = bilinear

    init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)
    return tf.get_variable(name="up_filter", initializer=init,shape=weights.shape)


def upscore_layer(bottom, name, shape, num_classes,ksize, stride):
    strides = [1, stride, stride, 1]
    with tf.variable_scope(name):
        infeatures = bottom.get_shape()[3].value
        weight_shape = [ksize,ksize,num_classes,infeatures]

        output_shape = tf.stack([shape[0], shape[1], shape[2], num_classes])

        weights = get_deconv_filter(weight_shape)
        deconv = tf.nn.conv2d_transpose(bottom,weights,output_shape,strides=strides,padding='SAME')

        return deconv

def conv_for_extra_layers(bottom, inchannels, outchannels, name, shape=[], isRelu=True):

    with tf.variable_scope(name):
        filt, conv_biases = get_filter_and_bias(name,shape)
        tf.summary.histogram('added_filters'+name, filt)
        
        conv = tf.nn.conv2d(bottom,filt,[1,1,1,1],padding='SAME')
        bias = tf.nn.bias_add(conv, conv_biases)
        if isRelu:
            relu = tf.nn.relu(bias)
        else:
            relu = bias
        return relu

def model(img):

    # Convert RGB to BGR
    tf.summary.image('train_images',img)
    red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=img)
        
    bgr = tf.concat(axis=3, values=[
        blue - VGG_MEAN[0],
        green - VGG_MEAN[1],
        red - VGG_MEAN[2],
        ])


    conv1_1 = conv_layer(bgr,3,64,'conv1_1')
    conv1_2 = conv_layer(conv1_1,64,64,'conv1_2')
    pool1 = tf.nn.max_pool(conv1_2,[1,2,2,1],[1,2,2,1],padding='SAME',name='pool1')

    conv2_1 = conv_layer(pool1,64,128,'conv2_1')
    conv2_2 = conv_layer(conv2_1,128,128,'conv2_2')
    pool2 = tf.nn.max_pool(conv2_2,[1,2,2,1],[1,2,2,1],padding='SAME',name='pool2') 

    conv3_1 = conv_layer(pool2,128,256,'conv3_1')
    conv3_2 = conv_layer(conv3_1,256,256,'conv3_2')
    conv3_3 = conv_layer(conv3_2,256,256,'conv3_3')
    pool3 = tf.nn.max_pool(conv3_3,[1,2,2,1],[1,2,2,1],padding='SAME',name='pool3')

    conv4_1 = conv_layer(pool3,256,512,'conv4_1')
    conv4_2 = conv_layer(conv4_1,512,512,'conv4_2')
    conv4_3 = conv_layer(conv4_2,512,512,'conv4_3')
    pool4 = tf.nn.max_pool(conv4_3,[1,2,2,1],[1,2,2,1],padding='SAME',name='pool4')

    conv5_1 = conv_for_extra_layers(pool4,512,256,'conv5_1', [3,3,512,256])
    conv5_2 = conv_for_extra_layers(conv5_1,256,256,'conv5_2', [3,3,256,256])
    conv5_3 = conv_for_extra_layers(conv5_2,256,256,'conv5_3', [3,3,256,256])
    pool5 = tf.nn.max_pool(conv5_3,[1,2,2,1],[1,2,2,1],padding='SAME',name='pool5')
    
    conv6_1 = conv_for_extra_layers(pool5,256,128,'conv6_1', [3,3,256,128])
    conv6_2 = conv_for_extra_layers(conv6_1,128,128,'conv6_2', [3,3,128,128])
    pool6 = tf.nn.max_pool(conv6_2,[1,2,2,1],[1,2,2,1],padding='SAME',name='pool6')
    conv9_1 = conv_for_extra_layers(pool6,128,num_classes,'conv9_1', [1,1,128,num_classes], False)
    pool9 = tf.nn.avg_pool(conv9_1,[1,4,4,1],[1,4,4,1],padding='SAME',name='pool8')
   
    pred = tf.argmax(pool9,axis=3)

    return pred, pool9, pool5




graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(tf.float32, shape = [None, None, None, 3])
    y = tf.placeholder(tf.int32, shape = [None])
    # labels = tf.one_hot(tf.cast(y, tf.int32),num_classes)
    pred, logits, pool5 = model(x)	
    pred = tf.squeeze(pred, axis=[1,2])
    # # print(logits.dtype)
    logits_ = tf.squeeze(logits, axis=[1,2])

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_,labels=y))
    # # loss = tf.reduce_mean(focal_loss(tf.reshape(labels,[-1,num_classes]), tf.reshape(logits,[-1,num_classes])))
    tf.summary.scalar('step_loss',loss)
    
    optimizer = tf.train.AdamOptimizer(learningRate).minimize(loss)
    
with tf.Session(graph=graph) as sess:
    saver = tf.train.Saver(max_to_keep=50)

    tf.global_variables_initializer().run()
    writer = tf.summary.FileWriter("outputLogs", sess.graph)
    saver.restore(sess,'modelNew/model-20')
    
    
    for epoch in range(n_epochs):
        for start, end in zip( \
            range(0, len(trainImages), batchSize),
            range(batchSize, len(trainImages), batchSize)):

            predictions, stepLoss, l,  opti, summary_ = sess.run([pred, loss, logits_, optimizer, summary], feed_dict = {
                    x: trainImages[start:end],
                    y: trainLabels[start:end]
                })
            writer.add_summary(summary_)
            print(stepLoss)
            # print(l)
            print(predictions)
            print(trainLabels[start:end])
        print ("Epoch ", epoch, " is done. Saving the model ... ")
        if epoch%5 == 0:
            saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)
        for start, end in zip( \
            range(0, len(valImages), batchSize),
            range(batchSize, len(valImages), batchSize)):

            predictions= sess.run([pred], feed_dict = {
                    x: valImages[start:end],
                    y: valLabels[start:end]
                })
            print(predictions)
            print(valLabels[start:end])
            accuracy = 100*np.sum(np.array(predictions) == np.array(valLabels[start:end]))/batchSize
            print('******************accuracy************************')
            print(accuracy)
            print('*********************************************')
            
         