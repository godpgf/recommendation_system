from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import os

GENRES = [
    'Action', 'Adventure', 'Animation', "Children", 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', "IMAX", 'Musical',
    'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]

AGES = [1, 18, 25, 35, 45, 50, 56]
OCCUPATION_LEN = 21
NUM_USER_IDS = 6040
NUM_ITEM_IDS = 3952


def _read_ratings_file(filename, sep="\t"):
    col_names = ["user_id", "item_id", "rating", "timestamp"]
    df = pd.read_csv(filename, sep=sep, header=None, names=col_names, engine='python')
    for col in ("user_id", "item_id"):
        df[col] = df[col].astype(np.int32)
    df["user_id"] -= 1
    df["item_id"] -= 1
    df["rating"] = df["rating"].astype(np.float32)
    return df


def _read_users_file(filename, sep="\t"):
    col_names = ["user_id", "gender", "age", "occupation", "zip-code"]
    df = pd.read_csv(filename, sep=sep, header=None, names=col_names, engine='python')

    f_v = np.array([0., 1.])
    m_v = np.array([1., 0.])
    df["gender"] = df["gender"].apply(lambda entry: m_v if entry == "M" else f_v)

    age_map = {}
    for id, age in enumerate(AGES):
        v = np.zeros(len(AGES))
        v[id] = 1
        age_map[age] = v
    df["age"] = df["age"].apply(lambda age: age_map[age])

    occupation_map = {}
    for id in range(OCCUPATION_LEN):
        v = np.zeros(OCCUPATION_LEN)
        v[id] = 1
        occupation_map[id] = v
    df["occupation"] = df["occupation"].apply(lambda ocp: occupation_map[ocp])

    for col in ["user_id"]:
        df[col] = df[col].astype(np.int32)
    df["user_id"] -= 1
    return df


def _read_movies_file(filename, sep="\t"):
    col_names = ["item_id", "titles", "genres"]
    df = pd.read_csv(filename, sep=sep, header=None, names=col_names, engine='python')
    for col in ["item_id"]:
        df[col] = df[col].astype(np.int32)
    df["item_id"] -= 1

    def _map_fn(entry):
        entry = entry.replace("Children's", "Children")  # naming difference.
        movie_genres = entry.split("|")
        output = np.zeros((len(GENRES),), dtype=np.int64)
        for i, genre in enumerate(GENRES):
            if genre in movie_genres:
                output[i] = 1.0
        return output

    df["genres"] = df["genres"].apply(_map_fn)
    return df.drop(columns=["titles"])


def read_file(path, sep="\t"):
    df_ratings = _read_ratings_file(os.path.join(path, "ratings.dat"), sep=sep)
    df_movies = _read_movies_file(os.path.join(path, "movies.dat"), sep=sep)
    df_users = _read_users_file(os.path.join(path, "users.dat"), sep=sep)

    df = df_ratings.merge(df_movies, on="item_id").merge(df_users, on="user_id")

    rows = len(df)
    # Purely integer-location based indexing for selection by position
    df = df.iloc[np.random.permutation(rows)].reset_index(drop=True)
    # Separate data into train and test, 90% for train and 10% for test
    split_index = int(rows * 0.9)
    # Use indices to separate the data
    df_train = df[0:split_index]
    df_test = df[split_index:].reset_index(drop=True)
    return df_train, df_test


def read_score_file(path, sep="\t"):
    df = _read_ratings_file(os.path.join(path, "ratings.dat"), sep=sep)
    rows = len(df)
    # Purely integer-location based indexing for selection by position
    df = df.iloc[np.random.permutation(rows)].reset_index(drop=True)
    # for col in ["user_id", "item_id", "rating"]:
    #     df[col] = df[col].astype(np.int32)
    # Separate data into train and test, 90% for train and 10% for test
    split_index = int(rows * 0.9)
    # Use indices to separate the data
    df_train = df[0:split_index]
    df_test = df[split_index:].reset_index(drop=True)
    return df_train, df_test


def df_2_dic(df):
    dic = {}
    for index, row in df.iterrows():
        user, item, record = int(row["user_id"]), int(row["item_id"]), row["rating"]
        dic.setdefault(user, {})
        dic[user][item] = record
    return dic


def df_2_dic_all(df):
    rating_dic = {}
    movie_dic = {}
    user_dic = {}
    for index, row in df.iterrows():
        user, item, record = int(row["user_id"]), int(row["item_id"]), row["rating"]
        rating_dic.setdefault(user, {})
        rating_dic[user][item] = record

        if item not in movie_dic:
            movie_dic[item] = row["genres"]

        if user not in user_dic:
            user_dic[user] = (row["age"], row["gender"], row["occupation"])

    return rating_dic, user_dic, movie_dic


def add_negative_sample(user_item_dic, sorted_item, ratio):
    data = []
    for user, items in user_item_dic.items():
        # dic.setdefault(user, {})
        for item, rating in items.items():
            # dic[item] = 1.0
            data.append([user, item, 1.0])
        neg_size = ratio * len(items)
        cnt = 0
        for i in range(len(sorted_item)):
            item, pop = sorted_item[i]
            if item not in items:
                # dic[item] = 0.0
                data.append([user, item, 0.0])
                cnt += 1
            if cnt >= neg_size:
                break
    data = np.array(data)
    np.random.shuffle(data)
    return data



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
            # modify by yanyu 给流行度加上权重，一个用户看到的电影越少，他的意见权重越高
            # item_popularity[i] += 1
            item_popularity[i] += 1.0 / len(items)
    return item_popularity


class ShuffleIterator(object):
    """
    Randomly generate batches
    """
    def __init__(self, inputs, batch_size=10):
        self.inputs = []
        for data_array in inputs:
            # print(data_array.values)
            if isinstance(data_array.values[0], np.int32) or isinstance(data_array.values[0], np.float32):
                self.inputs.append(data_array.values)
            else:
                self.inputs.append(np.array([[d for d in data] for data in data_array.values]))
                # print("dddd")
            # self.inputs.append(np.array([data for data in data_array.values]))
        self.batch_size = batch_size
        self.num_cols = len(self.inputs)
        self.len = len(self.inputs[0])
        # self.inputs = np.transpose(np.vstack([np.array(self.inputs[i]) for i in range(self.num_cols)]))

    def __len__(self):
        return self.len

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        ids = np.random.randint(0, self.len, (self.batch_size,))
        #out = self.inputs[ids, :]
        #return [out[:, i] for i in range(self.num_cols)]
        return [self.inputs[i][ids] for i in range(self.num_cols)]


class OneEpochIterator(ShuffleIterator):
    """
    Sequentially generate one-epoch batches, typically for test data
    """
    def __init__(self, inputs, batch_size=10):
        super(OneEpochIterator, self).__init__(inputs, batch_size=batch_size)
        if batch_size > 0:
            self.idx_group = np.array_split(np.arange(self.len), np.ceil(self.len / batch_size))
        else:
            self.idx_group = [np.arange(self.len)]
        self.group_id = 0

    def next(self):
        if self.group_id >= len(self.idx_group):
            self.group_id = 0
            raise StopIteration
        ids = self.idx_group[self.group_id]
        # out = self.inputs[self.idx_group[self.group_id], :]
        self.group_id += 1
        # return [out[:, i] for i in range(self.num_cols)]
        return [self.inputs[i][ids] for i in range(self.num_cols)]