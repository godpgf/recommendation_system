# Imports for data io operations
from collections import deque
from six import next
import onehot_reader as readers

# Main imports for training
import tensorflow as tf
import numpy as np

# Evaluate train times per epoch
import time

# Constant seed for replicating training results
np.random.seed(42)

EMBEDDING_SIZE = 6
BATCH_SIZE = 1000  # Number of samples per batch
MAX_EPOCHS = 1000  # Number of times the network sees all the training data

feat_index = tf.placeholder(dtype=tf.int32, shape=[None, readers.FIELD_SIZE], name='feat_index') # [None, field_size]
feat_value = tf.placeholder(dtype=tf.float32, shape=[None, readers.FIELD_SIZE], name='feat_value') # [None, field_size]
label = tf.placeholder(dtype=tf.float32, shape=[None,1], name='label')


def clip(x):
    return np.clip(x, 1.0, 5.0)


def model(feat_index, feat_value, dnn_hidden_units=[64, 64, 16], device="/cpu:0"):
    with tf.device(device):
        with tf.variable_scope('lsi', reuse=True):
            # 记录onehot的稠密编码向量
            v_embedding = tf.Variable(
                initial_value=tf.random_normal(shape=[readers.FEATURE_SIZE, EMBEDDING_SIZE], mean=0, stddev=0.1),
                name='feature_embedding',
                dtype=tf.float32)

            # Sparse Features -> Dense Embedding
            embeddings_origin = tf.nn.embedding_lookup(v_embedding,
                                                       ids=feat_index)  # [None, field_size, embedding_size]
            feat_value_reshape = tf.reshape(tensor=feat_value, shape=[-1, readers.FIELD_SIZE, 1])  # [None, field_size, 1]

            # 记录onehot的直接权重
            w_bias = tf.Variable(initial_value=tf.random_uniform(shape=[readers.FEATURE_SIZE, 1],minval=0.0,maxval=1.0),
                                      name='w_bias',
                                      dtype=tf.float32)

            # --------- 一维特征 -----------
            y_first_order = tf.nn.embedding_lookup(w_bias, ids=feat_index)  # [None, field_size, 1]
            w_mul_x = tf.multiply(y_first_order, feat_value_reshape)  # [None, field_size, 1]  Wi * Xi
            y_first_order = tf.reduce_sum(input_tensor=w_mul_x, axis=2)  # [None, field_size]

            # --------- 二维组合特征 ----------
            embeddings = tf.multiply(embeddings_origin,
                                     feat_value_reshape)  # [None, field_size, embedding_size] multiply不是矩阵相乘，而是矩阵对应位置相乘。这里应用了broadcast机制。

            # sum_square part 先sum，再square
            summed_features_emb = tf.reduce_sum(input_tensor=embeddings, axis=1)  # [None, embedding_size]
            summed_features_emb_square = tf.square(summed_features_emb)

            # square_sum part
            squared_features_emb = tf.square(embeddings)
            squared_features_emb_summed = tf.reduce_sum(input_tensor=squared_features_emb,
                                                        axis=1)  # [None, embedding_size]

            # second order
            y_second_order = 0.5 * tf.subtract(summed_features_emb_square, squared_features_emb_summed)

            # 构建deep模型----------------------------------------------------------------------------------------------------
            num_layer = len(dnn_hidden_units)
            input_size = readers.FIELD_SIZE * EMBEDDING_SIZE
            glorot = np.sqrt(2.0 / (input_size + dnn_hidden_units[0]))  # glorot_normal: stddev = sqrt(2/(fan_in + fan_out))

            # ----------- Deep Component ------------
            y_deep = tf.reshape(embeddings_origin, shape=[-1,
                                                          readers.FIELD_SIZE * EMBEDDING_SIZE])  # [None, field_size * embedding_size]
            reg = None
            for i in range(0, len(dnn_hidden_units)):
                layer = tf.Variable(
                    initial_value=tf.random_normal(shape=[input_size, dnn_hidden_units[i]], mean=0, stddev=glorot),
                    dtype=tf.float32)
                bias = tf.Variable(
                    initial_value=tf.random_normal(shape=[1, dnn_hidden_units[i]], mean=0, stddev=glorot),
                    dtype=tf.float32)
                y_deep = tf.add(tf.matmul(y_deep, layer), bias)
                y_deep = tf.nn.relu(y_deep)
                input_size = dnn_hidden_units[i]
                glorot = np.sqrt(2.0 / (dnn_hidden_units[i - 1] + dnn_hidden_units[i]))
                if reg is None:
                    reg = tf.nn.l2_loss(layer)
                else:
                    reg = tf.add(reg, tf.nn.l2_loss(layer))

            # Output Layer
            fm_size = readers.FIELD_SIZE + EMBEDDING_SIZE
            input_size += fm_size
            glorot = np.sqrt(2.0 / (input_size + 1))
            concat_projection = tf.Variable(
                initial_value=tf.random_normal(shape=[input_size, 1], mean=0, stddev=glorot),
                dtype=tf.float32)
            concat_bias = tf.Variable(tf.constant(value=0.01), dtype=tf.float32)
            reg = tf.add(reg, tf.nn.l2_loss(concat_projection))
            # ----------- output -----------
            concat_input = tf.concat([y_first_order, y_second_order, y_deep], axis=1)
            out = tf.add(tf.matmul(concat_input, concat_projection), concat_bias)
            out = tf.nn.relu(out)
    return out, reg


def loss(infer, reg, rate_batch, learning_rate=0.000006, lambda_reg=0.0,
         device="/cpu:0"):
    with tf.device(device):
        # Use L2 loss to compute penalty
        cost_l2 = tf.nn.l2_loss(tf.subtract(infer, rate_batch))
        penalty = tf.constant(lambda_reg, dtype=tf.float32, shape=[], name="l2")

        cost = cost_l2 + tf.multiply(reg, penalty)
        # cost = cost_l2 + tf.multiply(reg_wide, penalty_wide)
        # 'Follow the Regularized Leader' optimizer
        train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    return cost, train_op


df_train_index, df_train_value, train_rating, df_test_index, df_test_value, test_rating = readers.read_file("ml-1m", "::")


samples_per_batch = len(df_train_index) // BATCH_SIZE
print("Number of train samples %d, test samples %d, samples per batch %d" %
      (len(df_train_index), len(df_test_index), samples_per_batch))

# Peeking at the top 5 user values
print(df_train_index["user_id"].head())
print(df_test_index["user_id"].head())

# Peeking at the top 5 item values
print(df_train_index["item_id"].head())
print(df_test_index["item_id"].head())


# Using a shuffle iterator to generate random batches, for training
iter_train = readers.ShuffleIterator(df_train_index, df_train_value, train_rating,
                                     batch_size=BATCH_SIZE)

# Sequentially generate one-epoch batches, for testing
iter_test = readers.OneEpochIterator(df_test_index, df_test_value, test_rating,
                                     batch_size=-1)

infer, reg = model(feat_index, feat_value)
_, train_op = loss(infer, reg, label)


saver = tf.train.Saver()
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    print("%s\t%s\t%s\t%s" % ("Epoch", "Train Error", "Val Error", "Elapsed Time"))
    errors = deque(maxlen=samples_per_batch)
    start = time.time()
    for i in range(MAX_EPOCHS * samples_per_batch):
        index, value, rating = next(iter_train)
        _, pred_batch = sess.run([train_op, infer], feed_dict={feat_index:index, feat_value: value,
                                                               label: rating})
        pred_batch = clip(pred_batch)
        errors.append(np.power(pred_batch - rating, 2))
        if i % samples_per_batch == 0:
            train_err = np.sqrt(np.mean(errors))
            test_err2 = np.array([])
            for index, value, rating in iter_test:
                pred_batch = sess.run(infer, feed_dict={feat_index:index, feat_value: value,
                                                               label: rating})
                pred_batch = clip(pred_batch)
                test_err2 = np.append(test_err2, np.power(pred_batch - rating, 2))
            end = time.time()

            print("%02d\t%.3f\t\t%.3f\t\t%.3f secs" % (
                i // samples_per_batch, train_err, np.sqrt(np.mean(test_err2)), end - start))
            start = end
    saver.save(sess, './save/')