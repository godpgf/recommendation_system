import numpy as np
from readers import base_reader as reader
import math
from pprint import pprint

# Evaluate train times per epoch
import time

# Constant seed for replicating training results
np.random.seed(42)

df_train, df_test = reader.read_file("../data/ml-1m", "::")

# Peeking at the top 5 user values
print(df_train["user_id"].head())
print(df_test["user_id"].head())

# Peeking at the top 5 item values
print(df_train["item_id"].head())
print(df_test["item_id"].head())

# Peeking at the top 5 rate values
print(df_train["rating"].head())
print(df_test["rating"].head())


dic_train = reader.df_2_dic(df_train)
dic_test = reader.df_2_dic(df_test)


def eval_test(train, test, all_items, item_popularity, W, K, N):
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
    for user, tu in test.items():
        # tu = test.get(user, {})
        rank = recommender(user, train, W, K, N)
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


# 物品相似度
def itemSimilarity(train):
    # 两个物品之间的初步相似度
    cor_items = dict()
    # 物品被用户购买的次数
    n_users = dict()
    for u, items in train.items():
        for i in items:
            # 记录某个物品被选中的次数
            n_users[i] = n_users.get(i, 0) + 1
            for j in items:
                if i == j:
                    continue
                cor_items.setdefault(i, {})
                # 一个用户如果选过太多物品，这个人就不会太有个性
                cor_items[i][j] = cor_items[i].get(j, 0) + 1 / math.log(1 + len(items) * 1.0)

    W = dict()
    for i, related_items in cor_items.items():
        max_wi = 0.0
        for j, cij in related_items.items():
            W.setdefault(i, {})
            W[i][j] = cij / math.sqrt(n_users[i] * n_users[j])
            max_wi = max(W[i][j], max_wi)
        # 归一化
        for j, cij in related_items.items():
            W[i][j] /= max_wi
        W[i]["sort_items"] = sorted(W[i].items(), key=lambda c: c[1], reverse=True)
    return W


# 推荐
def recommender(user, train, W, K, N):
    """recommend to user N item according to K max similarity item
        给用户推荐K个物品，物品来源于与用户偏好物品的N个最相似的物品
    """
    rank = dict()
    interacted_items = train[user]
    for i, pi in interacted_items.items():
        # 找到和用户user已经打分的物品i最相似的k个物品，item_id和相似度是(j, wj)
        for j, wj in W[i]["sort_items"][0:K]:
            if j in interacted_items:
                continue
            # 得到用户对物品j的喜好预测
            rank[j] = rank.get(j, 0) + pi * wj
    # 从所有物品中取出n个
    return dict(sorted(rank.items(), key=lambda c: c[1], reverse=True)[0:N])


W = itemSimilarity(dic_train)
rank = recommender(344, dic_train, W, 5, 10)
print("测试给id为344的用户推荐10部电影：")
pprint(rank)
# 打开文件,清空内容
result = open('result_ibcf.data', 'w')
print(u'不同K值下推荐算法的各项指标(精度、召回率、覆盖率、流行度)\n')

all_items = reader.get_all_itens(dic_test)
print("完成所有物品统计")
item_popularity = reader.get_item_popularity(dic_train)
print("完成物品流行度统计")

print('K\t\tprecision\trecall\t\tCoverage\tPopularity')

for k in [5, 10, 20, 40, 820, 160]:
    pre, rec, cov, pop = eval_test(dic_train, dic_test, all_items, item_popularity, W, k, 10)
    print("%3d\t\t%.2f%%\t\t%.2f%%\t\t%.2f%%\t\t%.6f" % (k, pre * 100, rec * 100, cov * 100, pop))
    result.write(str(k) + ' ' + str('%2.2f' % (pre * 100)) + ' ' + str('%2.2f' % (rec * 100)) + ' ' + str(
        '%2.2f' % (cov * 100)) + ' ' + str('%2.6f' % pop) + '\n')
