# Imports for data io operations
from collections import deque
from six import next
import readers

# Main imports for training
import tensorflow as tf
import numpy as np

# Evaluate train times per epoch
import time

# Constant seed for replicating training results
np.random.seed(42)


USER_EMBEDDING_DIM = 4  # 16
ITEM_EMBEDDING_DIM = 8  # 64

BATCH_SIZE = 1000  # Number of samples per batch
MAX_EPOCHS = 1000  # Number of times the network sees all the training data

# 用户id的输入
user_batch = tf.placeholder(tf.int32, shape=[None], name="user_id")
# 物品id的输入
item_batch = tf.placeholder(tf.int32, shape=[None], name="item_id")
# 打分输入
rating_batch = tf.placeholder(tf.float32, shape=[None])
# 性别输入
gender_batch = tf.placeholder(tf.float32, shape=[None, 2])
# 物品标签输入
genres_batch = tf.placeholder(tf.float32, shape=[None, len(readers.GENRES)])
# 年龄输入
age_batch = tf.placeholder(tf.float32, shape=[None, len(readers.AGES)])
# 职业输入
occupation_batch = tf.placeholder(tf.float32, shape=[None, readers.OCCUPATION_LEN])


def clip(x):
    return np.clip(x, 1.0, 5.0)



def model(user_batch, item_batch, user_num, item_num, gender_batch, genres_batch, age_batch, occupation_batch,
          dnn_hidden_units=[24, 24, 6], device="/cpu:0"):
    with tf.device(device):
        with tf.variable_scope('lsi', reuse=True):
            # 创建全局偏置
            bias_global = tf.Variable(tf.constant(0.01, shape=[1]))

            # 深度用户隐藏向量
            w_user_deep = tf.Variable(tf.truncated_normal(shape=[user_num, USER_EMBEDDING_DIM], stddev=0.02))
            # 深度物品隐藏向量
            w_item_deep = tf.Variable(tf.truncated_normal(shape=[item_num, ITEM_EMBEDDING_DIM], stddev=0.02))
            # 通过用户的id取出对应的用户隐藏向量
            embd_user_deep = tf.nn.embedding_lookup(w_user_deep, user_batch)
            # 通过物品id取出对应的隐藏向量
            embd_item_deep = tf.nn.embedding_lookup(w_item_deep, item_batch)
            # tf.concat([embd_user_deep, embd_item_deep, gender_batch, genres_batch, age_batch], 1)

    with tf.device(device):
        # 得到深度模型
        x = tf.concat([embd_user_deep, embd_item_deep, gender_batch, genres_batch, age_batch, occupation_batch], 1)
        input_size = USER_EMBEDDING_DIM + ITEM_EMBEDDING_DIM + len(readers.GENRES) + 2 + len(
            readers.AGES) + readers.OCCUPATION_LEN

        reg_deep = tf.add(tf.nn.l2_loss(embd_user_deep), tf.nn.l2_loss(embd_item_deep))
        for hide_size in dnn_hidden_units:
            w = tf.Variable(tf.random_normal([input_size, hide_size], stddev=0.02))
            b = tf.Variable(tf.constant(0.0), [hide_size])
            x = tf.nn.relu(tf.matmul(x, w) + b)
            input_size = hide_size
            reg_deep = tf.add(reg_deep, tf.nn.l2_loss(w))
        deep = tf.reduce_sum(x, 1)

        infer = tf.nn.relu(deep + bias_global)
    return infer, reg_deep


def loss(infer, reg_deep, rate_batch, learning_rate=0.000006, lambda_deep=0.0,
         device="/cpu:0"):
    with tf.device(device):
        # Use L2 loss to compute penalty
        cost_l2 = tf.nn.l2_loss(tf.subtract(infer, rate_batch))
        penalty_deep = tf.constant(lambda_deep, dtype=tf.float32, shape=[], name="l2")
        cost = cost_l2 + tf.multiply(reg_deep, penalty_deep)
        # 'Follow the Regularized Leader' optimizer
        train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    return cost, train_op


df_train, df_test = readers.read_file("ml-1m", "::")
samples_per_batch = len(df_train) // BATCH_SIZE
print("Number of train samples %d, test samples %d, samples per batch %d" %
      (len(df_train), len(df_test), samples_per_batch))

# Peeking at the top 5 user values
print(df_train["user_id"].head())
print(df_test["user_id"].head())

# Peeking at the top 5 item values
print(df_train["item_id"].head())
print(df_test["item_id"].head())

# Peeking at the top 5 rate values
print(df_train["rating"].head())
print(df_test["rating"].head())

# Using a shuffle iterator to generate random batches, for training
iter_train = readers.ShuffleIterator([df_train["user_id"],
                                      df_train["item_id"],
                                      df_train["gender"],
                                      df_train["genres"],
                                      df_train["age"],
                                      df_train["occupation"],
                                      df_train['rating']],
                                     batch_size=BATCH_SIZE)

# Sequentially generate one-epoch batches, for testing
iter_test = readers.OneEpochIterator([df_test["user_id"],
                                      df_test["item_id"],
                                      df_test["gender"],
                                      df_test["genres"],
                                      df_test["age"],
                                      df_test["occupation"],
                                      df_test['rating']],
                                     batch_size=-1)

infer, reg_deep = model(user_batch, item_batch, readers.NUM_USER_IDS, readers.NUM_ITEM_IDS, gender_batch, genres_batch,
                        age_batch, occupation_batch)
_, train_op = loss(infer, reg_deep, rating_batch)

saver = tf.train.Saver()
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    print("%s\t%s\t%s\t%s" % ("Epoch", "Train Error", "Val Error", "Elapsed Time"))
    errors = deque(maxlen=samples_per_batch)
    start = time.time()
    for i in range(MAX_EPOCHS * samples_per_batch):
        user_id, item_id, gender, genres, age, occupation, rating = next(iter_train)
        _, pred_batch = sess.run([train_op, infer], feed_dict={user_batch: user_id,
                                                               item_batch: item_id,
                                                               gender_batch: gender,
                                                               genres_batch: genres,
                                                               age_batch: age,
                                                               occupation_batch: occupation,
                                                               rating_batch: rating})
        pred_batch = clip(pred_batch)
        errors.append(np.power(pred_batch - rating, 2))
        if i % samples_per_batch == 0:
            train_err = np.sqrt(np.mean(errors))
            test_err2 = np.array([])
            for user_id, item_id, gender, genres, age, occupation, rating in iter_test:
                pred_batch = sess.run(infer, feed_dict={user_batch: user_id,
                                                        item_batch: item_id,
                                                        gender_batch: gender,
                                                        genres_batch: genres,
                                                        age_batch: age,
                                                        occupation_batch: occupation,
                                                        rating_batch: rating})
                pred_batch = clip(pred_batch)
                test_err2 = np.append(test_err2, np.power(pred_batch - rating, 2))
            end = time.time()

            print("%02d\t%.3f\t\t%.3f\t\t%.3f secs" % (
                i // samples_per_batch, train_err, np.sqrt(np.mean(test_err2)), end - start))
            start = end
    saver.save(sess, './save/')
