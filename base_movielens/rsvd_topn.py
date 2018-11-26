import numpy as np
from readers import base_reader as reader
import math


class RSVD(object):
    def __init__(self, user_num, item_num, lr=0.01, reg=0.02, factor_num=10, score_range=(1.0, 5.0)):
        self.factor_num = factor_num
        self.user_num = user_num
        self.item_num = item_num
        self.score_range = score_range
        self.lr = lr
        self.reg = reg
        # 平均打分
        self.avg_score = None
        self.bu = np.zeros(self.user_num)
        self.bi = np.zeros(self.item_num)
        temp = np.sqrt(self.factor_num)
        self.pu = np.array(
            [[(0.1 * np.random.random() / temp) for _ in range(self.factor_num)] for j in range(self.user_num)])
        self.qi = np.array(
            [[0.1 * np.random.random() / temp for _ in range(self.factor_num)] for j in range(self.item_num)])

    def predict(self, user, item_list):
        ppscore_list = np.array([self._predict(self.bu[user], self.bi[item], self.pu[user], self.qi[item]) for item in item_list])
        return ppscore_list

    def _predict(self, bu, bi, pu, qi):
        pscore = self.avg_score + bu + bi + np.dot(pu, qi)
        if self.score_range is not None:
            if pscore < self.score_range[0]:
                pscore = self.score_range[0]
            elif pscore > self.score_range[1]:
                pscore = self.score_range[1]
        return pscore

    def train(self, uis_list):
        if self.avg_score is None:
            score = [uis[2] for uis in uis_list]
            self.avg_score = np.mean(score)

        for uis in uis_list:
            user = int(uis[0])
            item = int(uis[1])
            score = int(uis[2])
            pscore = self._predict(self.bu[user], self.bi[item], self.pu[user], self.qi[item])
            eui = score - pscore
            self.bu[user] += self.lr * (eui - self.reg * self.bu[user])
            self.bi[item] += self.lr * (eui - self.reg * self.bi[item])
            temp = self.pu[user]
            self.pu[user] += self.lr * (eui * self.qi[item] - self.reg * self.pu[user])
            self.qi[item] += self.lr * (temp * eui - self.reg * self.qi[item])

    def eval(self, uis_train, uis_test, epoch_num=100):
        pre_rmse = 10000.0
        for _ in range(epoch_num):
            self.train(uis_train)
            cur_rmse = self.test(uis_test)
            print("Iteration %d times,RMSE is : %f" % (_+1,cur_rmse))
            if cur_rmse > pre_rmse:
                print("The best RMSE is : %f" % (pre_rmse))
                break
            else:
                pre_rmse = cur_rmse

    def test(self, uis_list):
        rmse = 0.0
        for uis in uis_list:
            user = int(uis[0])
            item = int(uis[1])
            score = int(uis[2])
            pscore = self._predict(self.bu[user], self.bi[item], self.pu[user], self.qi[item])
            rmse += (score - pscore) ** 2
        return np.sqrt(rmse/len(uis_list))

# Constant seed for replicating training results
np.random.seed(42)

df_train, df_test = reader.read_score_file("../data/ml-1m", "::")
# uis_train, uis_test = df_train[["user_id", "item_id", "rating"]].values, df_test[["user_id", "item_id", "rating"]].values

dic_train = reader.df_2_dic(df_train)
dic_test = reader.df_2_dic(df_test)
item_popularity = reader.get_item_popularity(dic_train)
sorted_item = sorted(item_popularity.items(), key=lambda x: x[1], reverse=True)

# svd = RSVD(reader.NUM_USER_IDS, reader.NUM_ITEM_IDS)
# svd.eval(uis_train, uis_test)


def test_eval_items(svd, dic_train, dic_test, item_popularity, N=10):
    # dic_train = reader.df_2_dic(df_train)
    # dic_test = reader.df_2_dic(df_test)
    all_items = np.array(list(reader.get_all_itens(dic_test)))
    # item_popularity = reader.get_item_popularity(dic_train)

    # 推荐
    def recommender(user, N):

        interacted_items = dic_train[user]
        pred_batch = svd.predict(user, all_items)
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


print('ratio\t\tprecision\trecall\t\tCoverage\tPopularity')
for ratio in [1.0, 2.0, 3.0, 5.0, 10.0, 20.0]:
    svd_train = reader.add_negative_sample(dic_train, sorted_item, ratio)
    svd = RSVD(reader.NUM_USER_IDS, reader.NUM_ITEM_IDS)
    for i in range(20):
        svd.train(svd_train)
    pre, rec, cov, pop = test_eval_items(svd, dic_train, dic_test, item_popularity,10)
    print("%.1f\t\t%.2f%%\t\t%.2f%%\t\t%.2f%%\t\t%.6f" % (ratio, pre * 100, rec * 100, cov * 100, pop))
