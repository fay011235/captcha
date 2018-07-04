# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 11:20:59 2018

@author: fay
"""
import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from PIL import Image
import shutil

# 模型和样本路径的设置
# inception-V3瓶颈层节点个数
BOTTLENECK_TENSOR_SIZE = 2048
# 瓶颈层tenbsor name
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
# 图像输入tensor name
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'

# v3 path
MODEL_DIR = '/home/lcf/HKM/transfer_learn/inception_dec_2015/'  # './datasets/inception_dec_2015'
# v3 modefile
MODEL_FILE = 'tensorflow_inception_graph.pb'

# 特征向量 save path
CACHE_DIR = '/home/lcf/HKM/transfer_learn/bottleneck/'
# 数据path
#INPUT_DATA
train_data  = ['/home/lcf/HKM/captcha/change_label_1/','/home/lcf/HKM/captcha/change_label_2/','/home/lcf/HKM/captcha/labeled_1/','/home/lcf/HKM/captcha/labeled_2/']
valide_data = ['/home/lcf/HKM/captcha/label_10/','/home/lcf/HKM/captcha/label_20/']
# 模型存放地址
log_path = '/home/lcf/HKM/transfer_learn/log1/'
# 验证数据 percentage
VALIDATION_PERCENTAGE = 50
# 测试数据 percentage
TEST_PERCENTAGE = 50

# 神经网络参数的设置
LEARNING_RATE = 0.1#0.01
STEPS = 1000000
BATCH = 100

num_chars=4
alpha_num_len=36

# 把样本中所有的图片列表并按训练、验证、测试数据分开
def create_image_lists(testing_percentage, validation_percentage):
    result = {}
    file_list = []
    training_images = []
    testing_images = []
    validation_images = []
    for file in train_data:
        file_glob = os.path.join(file,'*.png')  # os.path.join(INPUT_DATA, dir_name, '*.' + extension)
        file_list.extend(glob.glob(file_glob))
        for file_name in file_list:
            training_images.append(file_name)
    for file in  valide_data:
        for file_name in file_list:
            chance = np.random.randint(100)
            if chance < validation_percentage:
                validation_images.append(file_name)
            else:
                testing_images.append(file_name)
    label_name=''
    result[label_name] = {
        'dir': '',
        'training': training_images,
        'testing': testing_images,
        'validation': validation_images,
    }
    return result

# 函数通过类别名称、所属数据集和图片编号获取一张图片的地址
def get_image_path(image_lists, image_dir, index, label_name, category):
    base_name = image_lists[label_name][category][index]
    full_path = os.path.join(image_dir, base_name)
    return full_path


# 函数获取Inception-v3模型处理之后的特征向量的文件地址
def get_bottleneck_path(image_lists, index, label_name, category):
    return get_image_path(image_lists, CACHE_DIR, index, label_name, category) + '.txt'


# 函数使用加载的训练好的Inception-v3模型处理一张图片，得到这个图片的特征向量。
def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values


# 函数会先试图寻找已经计算且保存下来的特征向量，如果找不到则先计算这个特征向量，然后保存到文件
def get_or_create_bottleneck(sess, image_lists, label_name, index, category, jpeg_data_tensor, bottleneck_tensor):
    base_name=os.path.basename(image_lists[label_name][category][index])
    bottleneck_path = os.path.join(CACHE_DIR, base_name)+'.txt'

    if not os.path.exists(bottleneck_path):
        image_path = image_lists[label_name][category][index]
        image_data = gfile.FastGFile(image_path, 'rb').read()
        # image_data = np.array(list(gfile.FastGFile(image_path, 'rb').read()))
        # image_data = np.reshape(image_data,(1,image_data.shape[0]))
        bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)

        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)
    else:
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]

    return bottleneck_values


def text2vec(text):
    number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
                'V', 'W', 'X', 'Y', 'Z']
    char_set = number + ALPHABET  # 如果验证码长度小于4, '_'用来补齐
    CHAR_SET_LEN = len(char_set)
    MAX_CAPTCHA = 4
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


# 函数随机获取一个batch的图片作为训练数据
def get_random_cached_bottlenecks(sess, image_lists, how_many, category, jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    for _ in range(how_many):
        # label_index = random.randrange(n_classes)
        label_name = list(image_lists.keys())[0]
        train_len = len(image_lists[label_name][category])
        image_index = random.randrange(train_len)
        # image_lists[''][category][image_index]
        bottleneck = get_or_create_bottleneck(
            sess, image_lists, label_name, image_index, category, jpeg_data_tensor, bottleneck_tensor)
        base_name = os.path.basename(image_lists[label_name][category][image_index])
        text = base_name[:base_name.find('.')]
        try:
            ground_truth = text2vec(text)
        except:
            try:
                shutil.move(image_lists[label_name][category][image_index],'/home/lcf/HKM/captcha/no_use/'+base_name)
                image_lists = create_image_lists(TEST_PERCENTAGE, VALIDATION_PERCENTAGE)
                continue
            except:
                continue
        # ground_truth = np.zeros(n_classes, dtype=np.float32)
        # ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)
    return bottlenecks, ground_truths


# 获取全部的测试数据，并计算正确率
def get_test_bottlenecks(sess, image_lists,category, jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    # label_name_list = list(image_lists.keys())
    label_name = list(image_lists.keys())[0]
    #category = 'testing'
    for index, unused_base_name in enumerate(image_lists[label_name][category]):
        bottleneck = get_or_create_bottleneck(sess, image_lists, label_name, index, category, jpeg_data_tensor,
                                              bottleneck_tensor)
        base_name = os.path.basename(image_lists[label_name][category][index])
        text = base_name[:base_name.find('.')]
        try:
            ground_truth = text2vec(text)
        except:
            continue
        # ground_truth = np.zeros(n_classes, dtype=np.float32)
        # ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)
    ground_truths = np.array(ground_truths)
    return bottlenecks, ground_truths


def main():
    image_lists = create_image_lists(TEST_PERCENTAGE, VALIDATION_PERCENTAGE)
    n_classes = 36 * 4  # len(image_lists.keys())

    # 读取已经训练好的Inception-v3模型。
    with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
        f = gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb')
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(
            graph_def, return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])

    # 定义新的神经网络输入
    bottleneck_input = tf.placeholder(tf.float32, [None, BOTTLENECK_TENSOR_SIZE], name='BottleneckInputPlaceholder')
    ground_truth_input = tf.placeholder(tf.float32, [None, n_classes], name='GroundTruthInput')

    # 定义一层全链接层
    with tf.name_scope('final_training_ops'):
        weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, n_classes], stddev=0.001))
        biases = tf.Variable(tf.zeros([n_classes]))
        logits = tf.matmul(bottleneck_input, weights) + biases
        #final_tensor = tf.nn.softmax(logits)

    label = tf.cast(ground_truth_input, tf.int32)
    total_loss = tf.constant(0.0)  # tf.nn.softmax_cross_entropy_with_logits_v2
    for branch in range(num_chars):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=logits[:,(branch * alpha_num_len):((branch + 1) * alpha_num_len)], labels=label[:, (branch * alpha_num_len):((branch + 1) * alpha_num_len)],name='xentropy')  # labels=label[:,branch]
        cross_entropy = tf.reduce_mean(cross_entropy, name='xentropy_mean')  # sparse_softmax_cross_entropy_with_logits
        total_loss += cross_entropy
    total_loss /= num_chars
    cross_entropy_mean=total_loss
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy_mean)
    # 定义交叉熵损失函数。
    '''
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=ground_truth_input)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy_mean)
    #train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy_mean)
    '''
    def compute_acc():
        label = tf.cast(ground_truth_input, tf.int64)
        total_acc_in_char = tf.constant(0.0)
        acc_word = tf.constant(True, dtype=tf.bool, shape=(BATCH, 1)) #logits[0].get_shape().as_list()[1]
        for branch in range(num_chars):
            acc = tf.equal(tf.argmax(logits[:,(branch*alpha_num_len):((branch+1)*alpha_num_len)], 1), tf.argmax(label[:,(branch*alpha_num_len):((branch+1)*alpha_num_len)],1)) #label[:,branch]
            acc_word = tf.logical_and(acc_word, acc)
            acc = tf.reduce_mean(tf.cast(acc, tf.float32))
            total_acc_in_char += acc
        total_acc_in_char /= num_chars
        total_acc_in_word = tf.reduce_mean(tf.cast(acc_word, tf.float32))
        return total_acc_in_char, total_acc_in_word

    # 计算正确率。
    with tf.name_scope('evaluation'):
        #correct_prediction = tf.equal(tf.argmax(final_tensor, 1), tf.argmax(ground_truth_input, 1))
        #evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        total_acc_in_char, total_acc_in_word=compute_acc()
        action = 'testing' #'training continue'  # training\testing\training continue
        with tf.Session() as sess:
            # sess=tf.Session()
            init = tf.global_variables_initializer()
            sess.run(init)
            saver = tf.train.Saver()
            loss_summary = tf.summary.scalar('loss', cross_entropy_mean)
            evaluation_summary = tf.summary.scalar('loss2', total_acc_in_word)
            all_summary = tf.summary.merge([loss_summary,evaluation_summary])
            summary_writer = tf.summary.FileWriter(log_path, sess.graph)

            if (action == 'testing') | (action == 'training continue'):
                ckpt = tf.train.get_checkpoint_state(log_path)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
            if (action == 'training') | (action == 'training continue'):
                # 训练过程。
                validation_accuracy0 = 0
                #test_bottlenecks, test_ground_truth = get_test_bottlenecks(sess, image_lists, 'validation',jpeg_data_tensor, bottleneck_tensor)

                for i in range(STEPS):
                    train_bottlenecks, train_ground_truth = get_random_cached_bottlenecks(
                        sess, image_lists, BATCH, 'training', jpeg_data_tensor, bottleneck_tensor)
                    _,summary_str,loss_value=sess.run([train_step,all_summary,cross_entropy_mean],feed_dict={bottleneck_input: train_bottlenecks, ground_truth_input: train_ground_truth})
                    summary_writer.add_summary(summary_str,STEPS)
                    #print('loss:',loss_value)
                    if i % 100 == 0 or i + 1 == STEPS:
                        validation_bottlenecks, validation_ground_truth = get_random_cached_bottlenecks(
                            sess, image_lists, BATCH, 'validation', jpeg_data_tensor, bottleneck_tensor)
                        validation_accuracy_char, validation_accuracy_word= sess.run([total_acc_in_char, total_acc_in_word], feed_dict={
                            bottleneck_input: validation_bottlenecks, ground_truth_input: validation_ground_truth})
                        train_accuracy = sess.run(total_acc_in_word, feed_dict={
                            bottleneck_input: train_bottlenecks, ground_truth_input: train_ground_truth})
                        print(
                            'Step %d: Validation accuracy on random sampled %d examples  char= %.1f%%  word=%.1f%% train_accuracy = %.1f%%' %
                            (i, BATCH, validation_accuracy_char * 100,validation_accuracy_word*100, train_accuracy * 100),'loss:',loss_value)
                        if validation_accuracy_word > validation_accuracy0:
                            saver.save(sess, os.path.join(log_path, 'model.ckpt'), STEPS)
                            validation_accuracy0 = validation_accuracy_word
            # 在最后的测试数据上测试正确率。
            test_bottlenecks, test_ground_truth = get_test_bottlenecks(
                sess, image_lists,'testing',jpeg_data_tensor, bottleneck_tensor)
            test_accuracy_char,test_accuracy_word = sess.run([total_acc_in_char, total_acc_in_word], feed_dict={
                bottleneck_input: test_bottlenecks, ground_truth_input: test_ground_truth})
            print('Final test accuracy char= %.1f%%  word= %.1f%%' % (test_accuracy_char*100,test_accuracy_word * 100))

if __name__ == '__main__':
    main()



