import pandas as pd
import pickle

train_file = r"C:\Users\ziqiancui2\research_files\data\amazon-book\train.txt"
test_file = r"C:\Users\ziqiancui2\research_files\data\amazon-book\test.txt"
trainUniqueUsers, trainItem, trainUser = [], [], []
m_item = 0
n_user = 0
traindataSize = 0
res = {}
with open(train_file) as f:
    for l in f.readlines():
        if len(l) > 0:
            l = l.strip('\n').split(' ')
            items = list(map(str, [int(i) for i in l[1:]]))
            uid = str(l[0])
            cur = [uid] + items
            res[uid] = (items[:])

#             trainUniqueUsers.append(uid)
#             trainUser.extend([uid] * len(items))
#             trainItem.extend(items)
#             m_item = max(m_item, max(items))
#             n_user = max(n_user, uid)
#             traindataSize += len(items)


with open(test_file) as f:
    for l in f.readlines():
        l = l.strip('\n').split(' ')
        if len(l) > 2:
            items = list(map(str, [int(i) for i in l[1:]]))
            uid = str(l[0])
            res[uid] += items

# 计算数目
user_set = list(res.keys())
user_set_num = len(user_set)
item_set = set()
for key in res.keys():
    item_set = item_set | set(res[key])
item_set = list(item_set)
item_set_num = len(item_set)

with open(r"C:\Users\ziqiancui2\PycharmProjects\hyperbolic_rec\data\dataset\Amazon\Books\Books_user_records.pkl",
          'wb') as fo:  # 将数据写入pkl文件
    pickle.dump(res, fo)

with open(r"C:\Users\ziqiancui2\PycharmProjects\hyperbolic_rec\data\dataset\Amazon\Books\Books_user_mapping.pkl",
          'wb') as fo:  # 将数据写入pkl文件
    pickle.dump(user_set, fo)
with open(r"C:\Users\ziqiancui2\PycharmProjects\hyperbolic_rec\data\dataset\Amazon\Books\Books_item_mapping.pkl",
          'wb') as fo:  # 将数据写入pkl文件
    pickle.dump(item_set, fo)

# movieLens 20M
from collections import defaultdict
import pandas as pd

path = r"C:\Users\ziqiancui2\research_files\data\ml-20m\ratings.csv"
data = pd.read_csv(path)


def timeSlice(time_set):
    time_min = min(time_set)
    time_map = dict()
    for time in time_set:
        time_map[time] = int(round(float(time - time_min)))
    return time_map


def cleanAndsort(User, time_map):
    User_filted = dict()
    user_set = set()
    item_set = set()
    for user, items in User.items():
        user_set.add(user)
        User_filted[user] = items
        for item in items:
            item_set.add(item[0])
    user_map = dict()
    item_map = dict()
    for u, user in enumerate(user_set):
        user_map[user] = u + 1
    for i, item in enumerate(item_set):
        item_map[item] = i + 1

    for user, items in User_filted.items():
        User_filted[user] = sorted(items, key=lambda x: x[1])  # sort by time
    # new_item_list = list(item_map.values())

    User_res = dict()
    for user, items in User_filted.items():
        User_res[user_map[user]] = list(map(lambda x: [item_map[x[0]], time_map[x[1]]], items))

    time_max = set()
    for user, items in User_res.items():
        time_list = list(map(lambda x: x[1], items))
        time_diff = set()
        for i in range(len(time_list) - 1):
            if time_list[i + 1] - time_list[i] != 0:
                time_diff.add(time_list[i + 1] - time_list[i])
        if len(time_diff) == 0:
            time_scale = 1
        else:
            time_scale = min(time_diff)
        time_min = min(time_list)
        User_res[user] = list(map(lambda x: x[0], items))
    #         time_max.add(max(set(map(lambda x: x[1], User_res[user]))))

    return User_res, len(user_set), len(item_set)


def data_partition_sort(fname, filter_core=20):
    User1 = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}

    print('Preparing data...')
    f = open(fname, 'r')
    time_set = set()

    user_count = defaultdict(int)
    item_count = defaultdict(int)
    for line in f:
        u, i, _, _ = line.rstrip().split(',')
        if u == "userId":
            continue
        u = int(u)
        i = int(i)
        user_count[u] += 1
        item_count[i] += 1
    f.close()

    f = open(fname, 'r')
    for line in f:
        u, i, rating, timestamp = line.rstrip().split(',')
        if u == "userId":
            continue
        u = int(u)
        i = int(i)
        # cate is str
        if user_count[u] < filter_core or item_count[i] < filter_core:
            continue
        timestamp = float(timestamp)
        time_set.add(timestamp)
        User1[u].append([i, timestamp])
    f.close()

    # some user originally have more than 20 items, but when items less 20 are filted, they have less than 20 items
    User = defaultdict(list)
    for u in User1:
        if len(User1[u]) < filter_core:
            continue
        User[u] = User1[u]

    time_map = timeSlice(time_set)
    User, _, _ = cleanAndsort(User, time_map)

    for user in list(User):
        nfeedback = len(User[user])
        if nfeedback < 10:
            del User[user]

    itemset = set()
    userset = list(User.keys())
    usernum = len(userset)
    for key in User.keys():
        itemset = itemset | set(User[key])
    itemnum = len(itemset)
    itemset = list(itemset)

    print('Preparing done...')
    return [User, usernum, itemnum, userset, itemset]


user_records, usernum, itemnum, userset, itemset = data_partition_sort(path, filter_core=20)
