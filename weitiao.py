# -*- coding: utf-8 -*-
#https://github.com/caicloud/tensorflow-tutorial/blob/master/Deep_Learning_with_TensorFlow/1.4.0/Chapter06/2.2%20fine_tuning.ipynb
import glob
import os.path
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import tensorflow.contrib.slim as slim
import inception_preprocessing
# 加载通过TensorFlow-Slim定义好的inception_v3模型。
import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3
#from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
#inception_v3模型的默认输入尺寸
image_size = 299
# 处理好之后的数据文件。
INPUT_DATA = 'D:/work2017/HKM/code/incept_learning/data/' #'../../datasets/flower_processed_data.npy'

# 保存训练好的模型的路径。
TRAIN_FILE = 'D:/work2017/HKM/code/incept_learning/data/log_weitiao' #'train_dir/model'
# 谷歌提供的训练好的模型文件地址。因为GitHub无法保存大于100M的文件，所以
# 在运行时需要先自行从Google下载inception_v3.ckpt文件。
CKPT_FILE = 'D:/work2017/HKM/code/incept_learning/inception_resnet_v2_2016_08_30/inception_resnet_v2_2016_08_30.ckpt' #'../../datasets/inception_v3.ckpt'

# 定义训练中使用的参数。
LEARNING_RATE = 0.0001
STEPS = 300
BATCH = 32
N_CLASSES = 144 #5  ########################
 
# 不需要从谷歌训练好的模型中加载的参数。
CHECKPOINT_EXCLUDE_SCOPES = 'InceptionV3/Logits,InceptionV3/AuxLogits'
# 需要训练的网络层参数明层，在fine-tuning的过程中就是最后的全联接层。
TRAINABLE_SCOPES='InceptionV3/Logits,InceptionV3/AuxLogit'

#获取所有需要从谷歌训练好的模型中加载的参数
def get_tuned_variables():
    exclusions = [scope.strip() for scope in CHECKPOINT_EXCLUDE_SCOPES.split(',')]

    variables_to_restore = []
    # 枚举inception-v3模型中所有的参数，然后判断是否需要从加载列表中移除。
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
    return variables_to_restore

#获取所有需要训练的变量列表
def get_trainable_variables():    
    scopes = [scope.strip() for scope in TRAINABLE_SCOPES.split(',')]
    variables_to_train = []
    
    # 枚举所有需要训练的参数前缀，并通过这些前缀找到所有需要训练的参数。
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train

#将向量转化为tensor——batch尺寸
def batch_data(start,batch_size): 
	data=[]
	for i in range(batch_size):		
		raw_image=training_images[start+i]
		raw_image=np.reshape(raw_image,(20,64,3)) 
		raw_image1=tf.convert_to_tensor(raw_image,dtype=tf.float64)
		image = inception_preprocessing.preprocess_image(raw_image1, height=image_size, width=image_size, is_training=True)
		data.append(image)
	x1=tf.reshape(data,[batch_size,299,299,3])
	return x1

def batch_valid(start,batch_size): 
	data=[]
	for i in range(batch_size):		
		raw_image=validation_images[start+i]
		raw_image=np.reshape(raw_image,(20,64,3)) 
		raw_image1=tf.convert_to_tensor(raw_image,dtype=tf.float64)
		image = inception_preprocessing.preprocess_image(raw_image1, height=image_size, width=image_size, is_training=True)
		data.append(image)
	x1=tf.reshape(data,[batch_size,299,299,3])
	return x1

def main():
    # 加载预处理好的数据。
    training_images=np.load(INPUT_DATA+'train_data.npy')
    n_training_example = len(training_images)
    training_labels=np.load(INPUT_DATA+'train_label.npy')
    validation_images = np.load(INPUT_DATA+'test_data.npy')
    validation_labels = np.load(INPUT_DATA+'test_label.npy')
    
    valid_tensor=batch_valid(0,len(validation_images))
    #processed_data = np.load(INPUT_DATA)
    #training_images = processed_data[0]
    #n_training_example = len(training_images)
    #training_labels = processed_data[1]
    
    #validation_images = processed_data[2]
    #validation_labels = processed_data[3]
    
    #testing_images = processed_data[4]
    #testing_labels = processed_data[5]
    #print("%d training examples, %d validation examples and %d testing examples." % (
    #    n_training_example, len(validation_labels), len(testing_labels)))
    
    # 定义inception-v3的输入，images为输入图片，labels为每一张图片对应的标签。
    # tf.reset_default_graph()
    images = tf.placeholder(tf.float32, [None, 299, 299, 3], name='input_images') #[None, 299, 299, 3]
    labels = tf.placeholder(tf.int64, [None,N_CLASSES], name='labels') #[None]
    
    # 定义inception-v3模型。因为谷歌给出的只有模型参数取值，所以这里
    # 需要在这个代码中定义inception-v3的模型结构。虽然理论上需要区分训练和
    # 测试中使用到的模型，也就是说在测试时应该使用is_training=False，但是
    # 因为预先训练好的inception-v3模型中使用的batch normalization参数与
    # 新的数据会有出入，所以这里直接使用同一个模型来做测试。

    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits, _ = inception_v3.inception_v3(
            images, num_classes=N_CLASSES, is_training=True)
    
    trainable_variables = get_trainable_variables()
    # 定义损失函数和训练过程。
    tf.losses.softmax_cross_entropy(labels, logits, weights=1.0) #tf.one_hot(labels, N_CLASSES)
    total_loss = tf.losses.get_total_loss()
    train_step = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(total_loss)
    
    # 计算正确率。
    with tf.name_scope('evaluation'): 
        num_chars=4;alpha_num_len=36
        label = tf.cast(labels, tf.int64)
        total_acc_in_char = tf.constant(0.0)
        acc_word = tf.constant(True, dtype=tf.bool, shape=(BATCH, 1)) #logits[0].get_shape().as_list()[1]
        for branch in range(num_chars):
            acc = tf.equal(tf.argmax(logits[:,(branch*alpha_num_len):((branch+1)*alpha_num_len)], 1), tf.argmax(label[:,(branch*alpha_num_len):((branch+1)*alpha_num_len)],1)) #label[:,branch]
            acc_word = tf.logical_and(acc_word, acc)
            acc = tf.reduce_mean(tf.cast(acc, tf.float32))
            total_acc_in_char += acc
        total_acc_in_char /= num_chars
        total_acc_in_word = tf.reduce_mean(tf.cast(acc_word, tf.float32))
    '''
    with tf.name_scope('evaluation'):            
        correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    '''            
    # 定义加载Google训练好的Inception-v3模型的Saver。
    load_fn = slim.assign_from_checkpoint_fn(
      CKPT_FILE,
      get_tuned_variables(),
      ignore_missing_vars=True)
    
    # 定义保存新模型的Saver。
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        #sess=tf.Session()
        # 初始化没有加载进来的变量。
        init = tf.global_variables_initializer()
        sess.run(init)
        valid_data=sess.run(valid_tensor)
        # 加载谷歌已经训练好的模型。
        print('Loading tuned variables from %s' % CKPT_FILE)
        #load_fn(sess)
            
        start = 0
        end = BATCH
        for i in range(STEPS): 
            data_x=sess.run(batch_data(start,BATCH))
            _, loss = sess.run([train_step, total_loss], feed_dict={
                images: data_x,  #training_images[start:end] #batch_data(start,BATCH)
                labels: training_labels[start:end]})

            if i % 30 == 0 or i + 1 == STEPS:
                saver.save(sess, TRAIN_FILE, global_step=i)
                
                total_acc_in_char,total_acc_in_word = sess.run([total_acc_in_char,total_acc_in_word], feed_dict={ #evaluation_step
                    images: valid_data, labels: validation_labels})     #validation_images
                print('Step %d: Training loss is %.1f total_acc_in_char = %.1f%%  total_acc_in_word = %.1f%%' % (
                    i, loss, total_acc_in_char * 100.0, total_acc_in_word * 100.0))
                            
            start = end
            if start == n_training_example:
                start = 0
            
            end = start + BATCH
            if end > n_training_example: 
                end = n_training_example
            
        # 在最后的测试数据上测试正确率。
        test_accuracy = sess.run(evaluation_step, feed_dict={
            images: testing_images, labels: testing_labels})
        print('Final test accuracy = %.1f%%' % (test_accuracy * 100))






def text2vec(text):
	number=['0','1','2','3','4','5','6','7','8','9']
	ALPHABET=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
	char_set = number + ALPHABET # 如果验证码长度小于4, '_'用来补齐
	CHAR_SET_LEN = len(char_set)
	MAX_CAPTCHA=4
	text_len = len(text)
	# print(text_len,text)
	if text_len > MAX_CAPTCHA:
		print(text)
		raise ValueError('验证码最长4个字符')
	vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)

	def char2pos(c):
		k = ord(c) - 48
		if k > 9:
			k = ord(c) - 55
			if k > 35:
				raise ValueError('No Map')
		return k

	for i, c in enumerate(text):
		# print text
		idx = i * CHAR_SET_LEN + char2pos(c)
		# print i,CHAR_SET_LEN,char2pos(c),idx
		vector[idx] = 1
	return vector

















