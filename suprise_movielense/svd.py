from surprise import KNNBasic
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
from readers import base_reader as reader
import numpy as np
import math

# Constant seed for replicating training results
np.random.seed(42)

df_train, df_test = reader.read_score_file("../data/ml-1m", "::")
# for (uid, iid, r) in df_train[["user_id", "item_id", "rating"]].itertuples(index=False):
#     print(uid, iid, r)

trainset = Dataset.load_from_df(df_train[["user_id", "item_id", "rating"]], Reader(rating_scale=(1, 5))).build_full_trainset()
testset = df_test[["user_id", "item_id", "rating"]].itertuples(index=False)


# We'll use the famous SVD algorithm.
algo = KNNBasic()

# Train the algorithm on the trainset, and predict ratings for the testset
algo.fit(trainset)
predictions = algo.test(testset)

# Then compute RMSE
accuracy.rmse(predictions)


def test_eval_items(algo, df_train, df_test, N=10):
    dic_train = reader.df_2_dic(df_train)
    dic_test = reader.df_2_dic(df_test)
    all_items = np.array(list(reader.get_all_itens(dic_test)))
    item_popularity = reader.get_item_popularity(dic_train)

    def predict(user, all_items):
        return np.array([algo.predict(user, item)[3] for item in all_items])

    # 推荐
    def recommender(user, N):

        interacted_items = dic_train[user]
        pred_batch = predict(user, all_items)
        index = np.argsort(-pred_batch)
        rank = {}

        # test = ""
        for id in index:
            if all_items[id] not in interacted_items:
                rank[all_items[id]] = pred_batch[id]
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
    for user, tu in dic_test.items():
        # tu = test.get(user, {})
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
    return hit / (pre * 1.0), hit / (rec * 1.0), len(recommend_items) / (len(all_items) * 1.0), ret


print('precision\trecall\t\tCoverage\tPopularity')
pre, rec, cov, pop = test_eval_items(algo, df_train, df_test, 10)
print("%.2f%%\t\t%.2f%%\t\t%.2f%%\t\t%.6f" % (pre * 100, rec * 100, cov * 100, pop))