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

    df["gender"] = df["gender"].apply(lambda entry: 1 if entry == "M" else 2)

    age_map = {age: id for id, age in enumerate(AGES)}
    df["age"] = df["age"].apply(lambda age: age_map[age])

    for col in ["user_id", "age", "occupation"]:
        df[col] = df[col].astype(np.int32)
    df["user_id"] -= 1
    return df


def _read_movies_file(filename, sep="\t"):
    col_names = ["item_id", "titles", "genres"]
    df = pd.read_csv(filename, sep=sep, header=None, names=col_names, engine='python')

    def _map_fn(entry):
        return entry.replace("Children's", "Children")  # naming difference.
    genres = df["genres"].apply(_map_fn).values

    genres_map = {f:np.zeros(len(genres), dtype=np.int32) for f in GENRES}
    for id, g in enumerate(genres):
        genres_list = g.split('|')
        for tag in genres_list:
            genres_map[tag][id] = 1

    for col in ["item_id"]:
        df[col] = df[col].astype(np.int32)
    df["item_id"] -= 1

    df = pd.DataFrame(dict(**{"item_id":df["item_id"].values}, **genres_map))
    return df


def init_column_offset():
    column_offset = {}
    feature_size = 0
    field_size = 0

    column_offset["gender"] = feature_size
    feature_size += 2
    field_size += 1

    column_offset["age"] = feature_size
    feature_size += len(AGES)
    field_size += 1

    column_offset["occupation"] = feature_size
    feature_size += OCCUPATION_LEN
    field_size += 1

    for g in GENRES:
        column_offset[g] = feature_size
        feature_size += 2
        field_size += 1

    column_offset["user_id"] = feature_size
    feature_size += NUM_USER_IDS
    field_size += 1

    column_offset["item_id"] = feature_size
    feature_size += NUM_ITEM_IDS
    field_size += 1
    return column_offset, feature_size, field_size


COLUMN_OFFSET, FEATURE_SIZE, FIELD_SIZE = init_column_offset()


# 将数据分成全局下标和值
def _split_df_2_index_and_value(df):
    index = {k: np.zeros(len(df), np.int32) for k, _ in COLUMN_OFFSET.items()}
    value = {k: np.zeros(len(df), np.float32) for k, _ in COLUMN_OFFSET.items()}
    for k, v in COLUMN_OFFSET.items():
        vs = df[k].values
        for i in range(len(df)):
            index[k][i] = COLUMN_OFFSET[k] + int(vs[i])
            value[k][i] = 1.0
    return pd.DataFrame(index), pd.DataFrame(value)


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
    df_train_index, df_train_value = _split_df_2_index_and_value(df_train)
    df_test_index, df_test_value = _split_df_2_index_and_value(df_test)
    return df_train_index, df_train_value, df_train['rating'].values, df_test_index, df_test_value, df_test['rating'].values


class ShuffleIterator(object):
    """
    Randomly generate batches
    """
    def __init__(self, df_index, df_value, rating, batch_size=10):
        self.input_index = np.zeros([len(COLUMN_OFFSET), len(df_index)], np.int32)
        self.input_value = np.zeros([len(COLUMN_OFFSET), len(df_value)], np.float32)
        self.rating = rating.reshape((-1,1))

        cur_col = 0
        for k, v in COLUMN_OFFSET.items():
            self.input_index[cur_col][:] = df_index[k].values
            cur_col += 1
        self.input_index = self.input_index.T

        cur_col = 0
        for k, v in COLUMN_OFFSET.items():
            self.input_value[cur_col][:] = df_value[k].values
            cur_col += 1
        self.input_value = self.input_value.T
        # for index, row in df_index.iterrows():
        #     cur_col = 0
        #     for k, v in COLUMN_OFFSET.items():
        #         self.input_index[index][cur_col] = row[k]
        #         cur_col += 1
        #
        # for index, row in df_value.iterrows():
        #     cur_col = 0
        #     for k, v in COLUMN_OFFSET.items():
        #         self.input_index[index][cur_col] = row[k]
        #         cur_col += 1

        self.batch_size = batch_size
        self.num_cols = len(COLUMN_OFFSET)
        self.len = len(df_index)
        # self.inputs = np.transpose(np.vstack([np.array(self.inputs[i]) for i in range(self.num_cols)]))

    def __len__(self):
        return self.len

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        ids = np.random.randint(0, self.len, (self.batch_size,))
        return self.input_index[ids], self.input_value[ids], self.rating[ids]


class OneEpochIterator(ShuffleIterator):
    """
    Sequentially generate one-epoch batches, typically for test data
    """
    def __init__(self, df_index, df_value, rating, batch_size=10):
        super(OneEpochIterator, self).__init__(df_index, df_value, rating, batch_size=batch_size)
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
        return self.input_index[ids], self.input_value[ids], self.rating[ids]