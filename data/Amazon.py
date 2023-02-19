from data.Dataset import DataSet


# Amazon review dataset
class Electronics(DataSet):
    def __init__(self):
        self.dir_path = './data/dataset/Amazon/Electronics/'
        self.user_record_file = 'Electronics_user_records.pkl'
        self.user_mapping_file = 'Electronics_user_mapping.pkl'
        self.item_mapping_file = 'Electronics_item_mapping.pkl'
        self.user_neighbors_file = 'user_neighbors_list.pkl'
        self.item_neighbors_file = 'item_neighbors_list.pkl'

        self.num_users = 40358
        self.num_items = 28147
        self.max_user_neighbors = 201
        self.max_item_neighbors = 124

        self.user_records = None
        self.user_mapping = None
        self.item_mapping = None

    def generate_dataset(self, seed=0, user_shift=0, item_shift=0):
        user_records = self.load_pickle(self.dir_path + self.user_record_file)
        user_mapping = self.load_pickle(self.dir_path + self.user_mapping_file)
        item_mapping = self.load_pickle(self.dir_path + self.item_mapping_file)
        user_neighbors_list = self.load_pickle(self.dir_path + self.user_neighbors_file)
        item_neighbors_list = self.load_pickle(self.dir_path + self.item_neighbors_file)

        assert self.num_users == len(user_mapping) and self.num_items == len(item_mapping)

        # split dataset
        train_set, test_set = self.split_data_randomly(user_records, seed)
        # train_set, test_set = self.split_data_sequentially(user_records)

        train_matrix = self.generate_csr_matrix(train_set, self.num_users + user_shift, self.num_items + item_shift,
                                                user_shift, item_shift)
        train_set = self.list_index_shift(train_set, increase_by=user_shift)
        test_set = self.list_index_shift(test_set, increase_by=item_shift)

        user_neighbor_matrix = self.generate_csr_matrix(user_neighbors_list, self.num_users + user_shift,
                                                        self.num_users + user_shift, user_shift, item_shift)
        item_neighbor_matrix = self.generate_csr_matrix(item_neighbors_list, self.num_items + item_shift,
                                                        self.num_items + item_shift, user_shift, item_shift)
        user_neighbors_list = self.list_index_shift(user_neighbors_list, increase_by=user_shift)
        item_neighbors_list = self.list_index_shift(item_neighbors_list, increase_by=item_shift)

        return train_set, train_matrix, test_set, user_neighbor_matrix, item_neighbor_matrix, self.max_user_neighbors, self.max_item_neighbors


class Books(DataSet):
    def __init__(self):
        self.dir_path = './data/dataset/Amazon/Books/'
        self.user_record_file = 'Books_user_records.pkl'
        self.user_mapping_file = 'Books_user_mapping.pkl'
        self.item_mapping_file = 'Books_item_mapping.pkl'
        # self.user_neighbors_file = 'user_neighbors_list.pkl'
        # self.item_neighbors_file = 'item_neighbors_list.pkl'

        self.num_users = 52643  # 77754
        self.num_items = 91599  # 66963
        # self.max_user_neighbors = 398
        # self.max_item_neighbors = 210

        self.user_records = None
        self.user_mapping = None
        self.item_mapping = None

    def generate_dataset(self, seed=0, user_shift=0, item_shift=0):
        user_records = self.load_pickle(self.dir_path + self.user_record_file)
        # print("user_records", user_records)
        user_mapping = self.load_pickle(self.dir_path + self.user_mapping_file)
        item_mapping = self.load_pickle(self.dir_path + self.item_mapping_file)

        assert self.num_users == len(user_mapping) and self.num_items == len(item_mapping)

        # split dataset
        inner_data_records, user_inverse_mapping, item_inverse_mapping = \
            self.convert_to_inner_index(user_records, user_mapping, item_mapping)
        # print("inner_data_records", inner_data_records)
        # split dataset
        train_set, test_set = self.split_data_randomly(inner_data_records, seed)
        # train_set, test_set = self.split_data_sequentially(user_records)
        self.num_users = self.num_users + user_shift
        self.num_items = self.num_items + item_shift
        train_matrix = self.generate_csr_matrix(train_set, self.num_users, self.num_items, user_shift, item_shift)
        train_set = self.list_index_shift(train_set, increase_by=user_shift)
        test_set = self.list_index_shift(test_set, increase_by=item_shift)

        return train_set, train_matrix, test_set


class CDs(DataSet):
    def __init__(self):
        self.dir_path = './data/dataset/Amazon/CDs/'
        self.user_record_file = 'CDs_user_records.pkl'
        self.user_mapping_file = 'CDs_user_mapping.pkl'
        self.item_mapping_file = 'CDs_item_mapping.pkl'
        # self.user_neighbors_file = 'user_neighbors_list.pkl'
        # self.item_neighbors_file = 'item_neighbors_list.pkl'

        self.num_users = 24934
        self.num_items = 24634
        # self.max_user_neighbors = 166
        # self.max_item_neighbors = 251

        self.user_records = None
        self.user_mapping = None
        self.item_mapping = None

    def generate_dataset(self, seed=0, user_shift=0, item_shift=0):
        user_records = self.load_pickle(self.dir_path + self.user_record_file)
        user_mapping = self.load_pickle(self.dir_path + self.user_mapping_file)
        item_mapping = self.load_pickle(self.dir_path + self.item_mapping_file)
        # user_neighbors_list = self.load_pickle(self.dir_path + self.user_neighbors_file)
        # item_neighbors_list = self.load_pickle(self.dir_path + self.item_neighbors_file)
        # print("user_records", user_records)
        # print("user_mapping", len(user_mapping))
        # print("item_mapping", len(item_mapping))

        assert self.num_users == len(user_mapping) and self.num_items == len(item_mapping)

        inner_data_records, user_inverse_mapping, item_inverse_mapping = \
            self.convert_to_inner_index(user_records, user_mapping, item_mapping)
        print("inner_data_records", inner_data_records)
        # split dataset
        train_set, test_set = self.split_data_randomly(inner_data_records, seed)
        # train_set, test_set = self.split_data_sequentially(user_records)
        self.num_users = self.num_users + user_shift
        self.num_items = self.num_items + item_shift
        train_matrix = self.generate_csr_matrix(train_set, self.num_users, self.num_items, user_shift, item_shift)
        train_set = self.list_index_shift(train_set, increase_by=user_shift)
        test_set = self.list_index_shift(test_set, increase_by=item_shift)

        # user_neighbor_matrix = self.generate_csr_matrix(user_neighbors_list, self.num_users + user_shift, self.num_users + user_shift, user_shift, item_shift)
        # item_neighbor_matrix = self.generate_csr_matrix(item_neighbors_list, self.num_items + item_shift, self.num_items + item_shift, user_shift, item_shift)
        # user_neighbors_list = self.list_index_shift(user_neighbors_list, increase_by=user_shift)
        # item_neighbors_list = self.list_index_shift(item_neighbors_list, increase_by=item_shift)

        return train_set, train_matrix, test_set


if __name__ == '__main__':
    data_set = CDs()
    train_set, train_matrix, test_set = data_set.generate_dataset(seed=0, user_shift=1, item_shift=1)
    print(len(train_set))
    print(len(test_set))
    print(train_set[0])
    print(test_set[0])
    print(train_set[-1])
    print(test_set[-1])
    print(train_matrix.shape)
    print(train_matrix.getrow(0))
    print(train_matrix.getcol(0).toarray())