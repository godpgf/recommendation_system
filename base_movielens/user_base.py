import numpy as np
import readers
import math
from pprint import pprint

# Evaluate train times per epoch
import time

# Constant seed for replicating training results
np.random.seed(42)

df_train, df_test = readers.read_file("ml-1m", "::")

# Peeking at the top 5 user values
print(df_train["user_id"].head())
print(df_test["user_id"].head())

# Peeking at the top 5 item values
print(df_train["item_id"].head())
print(df_test["item_id"].head())

# Peeking at the top 5 rate values
print(df_train["rating"].head())
print(df_test["rating"].head())


def df_2_dic(df):
    dic = {}
    for index, row in df.iterrows():
        user, item, record = row["user_id"], row["item_id"], row["rating"]
        dic.setdefault(user, {})
        dic[user][item] = record
    return dic


def userSimilarityBest(dic_train):
    """
    生成用户相似度矩阵-改进的
    推荐系统实践 p46
    """
    item_users = dict()
    for u, item in dic_train.items():
        for i in item.keys():
            item_users.setdefault(i, set())
            item_users[i].add(u)

    userSimBest = dict()

    # 计算用户间同时评分的物品
    user_item_count = dict()
    count = dict()
    for item, users in item_users.items():
        for u in users:
            user_item_count.setdefault(u, 0)
            user_item_count[u] += 1
            for v in users:
                if u == v: continue
                count.setdefault(u, {})
                count[u].setdefault(v, 0)
                # 用户u和用户v都买过这个商品item，所以相似度加1
                count[u][v] += 1
    # 计算相似度矩阵
    for u, related_users in count.items():
        userSimBest.setdefault(u, dict())
        for v, cuv in related_users.items():
            userSimBest[u][v] = cuv / math.sqrt(user_item_count[u] * user_item_count[v] * 1.0)
        userSimBest[u]["sort_items"] = sorted(userSimBest[u].items(), key=lambda x: x[1], reverse=True)
    return userSimBest


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


def get_all_itens(dic):
    all_items = set()
    for user, items in dic.items():
        for i in items.keys():
            all_items.add(i)
    return all_items


def get_item_popularity(dic):
    item_popularity = dict()
    # 计算物品流行度
    for user, items in dic.items():
        for i in items.keys():
            item_popularity.setdefault(i, 0)
            item_popularity[i] += 1
    return item_popularity


# 给用户推荐K个与之相似用户喜欢的物品
def recommender(user, train, W, K, N):
    rank = dict()
    interacted_items = train.get(user, {})
    for v, wuv in W[user]["sort_items"][0:K]:
        for i, rvi in train[v].items():
            if i in interacted_items:
                continue
            rank.setdefault(i, 0)
            rank[i] += wuv * rvi
    return dict(sorted(rank.items(), key=lambda x: x[1], reverse=True)[0:N])


dic_train = df_2_dic(df_train)
dic_test = df_2_dic(df_test)

W = userSimilarityBest(dic_train)
rank = recommender(344, dic_train, W, 5, 10)
print("测试给id为344的用户推荐10部电影：")
pprint(rank)
# 打开文件,清空内容
result = open('result_ubcf.data', 'w')
print(u'不同K值下推荐算法的各项指标(精度、召回率、覆盖率、流行度)\n')
print('K\t\tprecision\trecall\t\tCoverage\tPopularity')

all_items = get_all_itens(dic_test)
print("完成所有物品统计")
item_popularity = get_item_popularity(dic_train)
print("完成物品流行度统计")

for k in [5, 10, 20, 40, 80, 160]:
    pre, rec, cov, pop = eval_test(dic_train, dic_test, all_items, item_popularity, W, k, 10)
    print("%3d\t\t%.2f%%\t\t%.2f%%\t\t%.2f%%\t\t%.6f" % (k, pre * 100, rec * 100, cov * 100, pop))
    result.write(str(k) + ' ' + str('%2.2f' % (pre * 100)) + ' ' + str('%2.2f' % (rec * 100)) + ' ' + str(
        '%2.2f' % (cov * 100)) + ' ' + str('%2.6f' % pop) + '\n')
