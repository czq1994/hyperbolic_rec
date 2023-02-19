from data.Dataset import DataSet
import scipy


class ML20M(DataSet):
    def __init__(self):
        self.dir_path = './data/dataset/MovieLens/ml20m/'
        self.user_record_file = 'ml20m_item_sequences.pkl'
        self.user_mapping_file = 'ml20m_user_mapping.pkl'
        self.item_mapping_file = 'ml20m_item_mapping.pkl'
        # self.item_pair_count_file = 'ml20m_3_neighbors.npz'

        self.num_users = 129797
        self.num_items = 13649
        self.max_item_neighbors = 500

        self.user_records = None
        self.user_mapping = None
        self.item_mapping = None

    def generate_dataset(self, seed=0, user_shift=0, item_shift=0):
        user_records = self.load_pickle(self.dir_path + self.user_record_file)
        user_mapping = self.load_pickle(self.dir_path + self.user_mapping_file)
        item_mapping = self.load_pickle(self.dir_path + self.item_mapping_file)

        # item_pair_count = scipy.sparse.load_npz(self.dir_path + self.item_pair_count_file)
        #
        # item_neighbor_list = []
        # item_neighbor_counts = []
        # for i in range(item_pair_count.shape[0]):
        #     item_neighbor_list.append(item_pair_count[i].indices)
        #     item_neighbor_counts.append(item_pair_count[i].data)

        assert self.num_users == len(user_mapping) and self.num_items == len(item_mapping)

        # user_records = self.data_index_shift(user_records, increase_by=index_shift)

        inner_data_records, user_inverse_mapping, item_inverse_mapping = \
            self.convert_to_inner_index(user_records, user_mapping, item_mapping)
        # split dataset
        # train_set, test_set = self.split_data_randomly(inner_data_records, seed)
        train_set, test_set = self.split_data_sequentially1(inner_data_records)
        self.num_users = self.num_users + user_shift
        self.num_items = self.num_items + item_shift
        train_matrix = self.generate_csr_matrix(train_set, self.num_users, self.num_items, user_shift, item_shift)
        train_set = self.list_index_shift(train_set, increase_by=user_shift)
        test_set = self.list_index_shift(test_set, increase_by=item_shift)

        # split dataset
        # train_val_set, test_set = self.split_data_sequentially(user_records, test_radio=0.2)
        # train_set, val_set = self.split_data_sequentially(train_val_set, test_radio=0.1)

        # return train_set, val_set, train_val_set, test_set, item_neighbor_list, self.max_item_neighbors, \
        #        self.num_users, self.num_items + index_shift
        return train_set, train_matrix, test_set