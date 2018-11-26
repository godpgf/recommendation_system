# Imports for data io operations
from collections import deque
from six import next
from readers import base_reader as reader

# Main imports for training
import tensorflow as tf
import numpy as np
import math

# Evaluate train times per epoch
import time

# Constant seed for replicating training results
np.random.seed(42)

USER_EMBEDDING_DIM = 10
ITEM_EMBEDDING_DIM = 20
HIDE_EMBEDDING_DIM = 10

BATCH_SIZE = 4096  # Number of samples per batch
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
genres_batch = tf.placeholder(tf.float32, shape=[None, len(reader.GENRES)])
# 年龄输入
age_batch = tf.placeholder(tf.float32, shape=[None, len(reader.AGES)])
# 职业输入
occupation_batch = tf.placeholder(tf.float32, shape=[None, reader.OCCUPATION_LEN])


def clip(x):
    return np.clip(x, 1.0, 5.0)


def model(user_batch, item_batch, user_num, item_num, gender_batch, genres_batch, age_batch, occupation_batch,
          dnn_hidden_units=[64, 64, 16], device="/cpu:0"):
    with tf.device(device):
        with tf.variable_scope('lsi', reuse=True):
            # 创建全局偏置
            bias_global = tf.Variable(tf.constant(0.01, shape=[1]))

            # 创建广度用户偏置
            w_bias_user_wide = tf.Variable(tf.constant(0.01, shape=[user_num]))
            # 创建广度物品偏置
            w_bias_item_wide = tf.Variable(tf.constant(0.01, shape=[item_num]))
            # 通过用户的id取出对应的用户偏置
            bias_user_wide = tf.nn.embedding_lookup(w_bias_user_wide, user_batch)
            # 通过物品的id取出对应的物品偏置
            bias_item_wide = tf.nn.embedding_lookup(w_bias_item_wide, item_batch)
            # 创建用户隐语义向量
            w_user_wide = tf.Variable(tf.truncated_normal(shape=[user_num, HIDE_EMBEDDING_DIM], stddev=0.02))
            # 创建物品隐语义向量
            w_item_wide = tf.Variable(tf.truncated_normal(shape=[item_num, HIDE_EMBEDDING_DIM], stddev=0.02))
            # 通过用户的id取出对应的用户隐语义
            embd_user_wide = tf.nn.embedding_lookup(w_user_wide, user_batch)
            # 通过物品id取出对应的物品隐语义
            embd_item_wide = tf.nn.embedding_lookup(w_item_wide, item_batch)

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
        # 得到宽度模型
        wide = tf.reduce_sum(tf.multiply(embd_user_wide, embd_item_wide), 1) + bias_user_wide + bias_item_wide
        # 加入宽度模型的正则项
        reg_wide = tf.add(tf.nn.l2_loss(embd_user_wide), tf.nn.l2_loss(embd_item_wide))

        # 得到深度模型
        x = tf.concat([embd_user_deep, embd_item_deep, gender_batch, genres_batch, age_batch, occupation_batch], 1)
        input_size = USER_EMBEDDING_DIM + ITEM_EMBEDDING_DIM + len(reader.GENRES) + 2 + len(
            reader.AGES) + reader.OCCUPATION_LEN

        reg_deep = tf.add(tf.nn.l2_loss(embd_user_deep), tf.nn.l2_loss(embd_item_deep))
        for hide_size in dnn_hidden_units:
            w = tf.Variable(tf.random_normal([input_size, hide_size], stddev=0.01))
            b = tf.Variable(tf.constant(0.0), [hide_size])
            x = tf.nn.relu(tf.matmul(x, w) + b)
            input_size = hide_size
            reg_deep = tf.add(reg_deep, tf.nn.l2_loss(w))

        # concat_weight = tf.Variable(tf.random_normal([1+dnn_hidden_units[-1], 1], stddev=0.01))
        # wide = tf.reshape(tensor=wide, shape=(-1, 1))
        # infer = tf.reduce_sum(tf.nn.relu(tf.matmul(tf.concat([wide, x], 1), concat_weight) + bias_global), 1)

        deep = tf.reduce_sum(x, 1)
        infer = tf.nn.relu(wide + deep + bias_global)
    return infer, reg_wide, reg_deep


def loss(infer, reg_wide, reg_deep, rate_batch, learning_rate=0.0000032, lambda_wide=0.036, lambda_deep=0.001,
         device="/cpu:0"):
    with tf.device(device):
        # Use L2 loss to compute penalty
        cost_l2 = tf.nn.l2_loss(tf.subtract(infer, rate_batch))
        penalty_wide = tf.constant(lambda_wide, dtype=tf.float32, shape=[], name="l2")
        penalty_deep = tf.constant(lambda_deep, dtype=tf.float32, shape=[], name="l2")
        cost = cost_l2 + tf.multiply(reg_wide, penalty_wide) + tf.multiply(reg_deep, penalty_deep)
        # cost = cost_l2 + tf.multiply(reg_wide, penalty_wide)
        # 'Follow the Regularized Leader' optimizer

        global_step = tf.Variable(0, trainable=False)
        lr = tf.train.exponential_decay(learning_rate, global_step, MAX_EPOCHS * samples_per_batch, learning_rate * 160, staircase=True)

        train_op = tf.train.GradientDescentOptimizer(lr).minimize(cost)
    return cost, train_op


df_train, df_test = reader.read_file("../data/ml-1m", "::")
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
iter_train = reader.ShuffleIterator([df_train["user_id"],
                                     df_train["item_id"],
                                     df_train["gender"],
                                     df_train["genres"],
                                     df_train["age"],
                                     df_train["occupation"],
                                     df_train['rating']],
                                    batch_size=BATCH_SIZE)

# Sequentially generate one-epoch batches, for testing
iter_test = reader.OneEpochIterator([df_test["user_id"],
                                     df_test["item_id"],
                                     df_test["gender"],
                                     df_test["genres"],
                                     df_test["age"],
                                     df_test["occupation"],
                                     df_test['rating']],
                                    batch_size=-1)

infer, reg_wide, reg_deep = model(user_batch, item_batch, reader.NUM_USER_IDS, reader.NUM_ITEM_IDS, gender_batch,
                                  genres_batch,
                                  age_batch, occupation_batch)
_, train_op = loss(infer, reg_wide, reg_deep, rating_batch)

saver = tf.train.Saver()
init_op = tf.global_variables_initializer()


def test_eval_items(sess, user_batch, item_batch, gender_batch, genres_batch, age_batch, occupation_batch, pred, N=10):
    dic_train = reader.df_2_dic(df_train)
    dic_test_rating, dic_test_user, dic_test_movie = reader.df_2_dic_all(df_test)
    all_train_items = np.array(list(reader.get_all_itens(dic_train)))
    all_train_genres = np.array([dic_test_movie[item] for item in all_train_items])
    all_test_items = np.array(list(reader.get_all_itens(dic_test_rating)))
    item_popularity = reader.get_item_popularity(dic_train)

    # 推荐
    def recommender(user, N):
        all_users = np.empty(len(all_train_items), dtype=np.int32)
        all_users.fill(user)
        all_age = np.empty([len(all_train_items), len(reader.AGES)], dtype=np.float32)
        all_age.fill(dic_test_user[user][0])
        all_gender = np.empty([len(all_train_items), 2], dtype=np.float32)
        all_gender.fill(dic_test_user[user][1])
        all_occupation = np.empty([len(all_train_items), reader.OCCUPATION_LEN], dtype=np.float32)
        all_occupation.fill(dic_test_user[user][2])

        interacted_items = dic_train[user]
        pred_batch = sess.run(pred, feed_dict={user_batch: all_users, item_batch: all_train_items,
                                               gender_batch: all_gender, genres_batch: all_train_genres,
                                               age_batch: all_age, occupation_batch: all_occupation})
        index = np.argsort(-pred_batch)
        rank = {}

        # test = ""
        for id in index:
            if all_train_items[id] not in interacted_items:
                rank[all_train_items[id]] = pred_batch[id]
                # test += "%.4f "%pred_batch[id]
                if len(rank) >= N:
                    break
        # print(test)

        return rank

    """
    计算算法的精度和回调
    """
    hit = 0
    pre = 0
    rec = 0

    # 记录预测到的物品在总物品中的比重
    recommend_items = set()

    # 计算新鲜度:测评的最简单方法是利用推荐结果的平均流行度，越不热门的物品越可能让用户觉得新颖。返回值越小，新颖度越大
    ret = 0  # 新颖度结果
    n = 0  # 推荐的总个数

    # l = 0

    for user, tu in dic_test_rating.items():
        # tu = dic_test[user]
        rank = recommender(user, N)
        for item, pui in rank.items():
            if item in tu:
                hit += 1

            recommend_items.add(item)

            ret += math.log(1 + item_popularity[item])
            n += 1
        pre += N
        rec += len(tu)
        # l += 1
        # print("%d/%d"%(l, len(test)))
    ret /= n * 1.0
    # 精度=命中数/预测数，召回=命中数/总共评分数，覆盖率，
    return hit / (pre * 1.0), hit / (rec * 1.0), len(recommend_items) / (len(all_test_items) * 1.0), ret


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
    print('precision\trecall\t\tCoverage\tPopularity')
    pre, rec, cov, pop = test_eval_items(sess, user_batch, item_batch, gender_batch, genres_batch, age_batch,
                                         occupation_batch, infer, 10)
    print("%.2f%%\t\t%.2f%%\t\t%.2f%%\t\t%.6f" % (pre * 100, rec * 100, cov * 100, pop))
