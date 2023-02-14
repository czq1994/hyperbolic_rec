import numpy as np
from multiprocessing import Process, Queue


def negsamp_vectorized_bsearch_preverif(pos_inds, n_items, n_samp=32):
    """ Pre-verified with binary search
    `pos_inds` is assumed to be ordered
    reference: https://tech.hbc.com/2018-03-23-negative-sampling-in-numpy.html
    """
    raw_samp = np.random.randint(0, n_items - len(pos_inds), size=n_samp)
    pos_inds_adj = pos_inds - np.arange(len(pos_inds))
    neg_inds = raw_samp + np.searchsorted(pos_inds_adj, raw_samp, side='right')
    return neg_inds


def sample_function(train_matrix, pre_samples, batch_size, num_neg, result_queue, SEED):
    def sample():
        batch_pair_index = np.random.choice(pair_ids, size=batch_size)
        batch_train_pair = user_item_pairs[batch_pair_index]

        batch_user_id = batch_train_pair[:, 0]
        batch_item_id = batch_train_pair[:, 1]
        # random_u_neg_item = np.random.randint(user_neg_items.shape[1], size=num_neg)
        # random_u_neg_user = np.random.randint(user_neg_users.shape[1], size=num_first_order_neighbor)
        # random_i_neg_item = np.random.randint(item_neg_items.shape[1], size=num_first_order_neighbor)
        # random_u_pos_user = np.random.randint(user_pos_users.shape[1], size=num_first_order_neighbor)
        # random_i_pos_item = np.random.randint(item_pos_items.shape[1], size=num_first_order_neighbor)
        batch_index = np.arange(batch_size)
        neg_samples = user_neg_items[batch_user_id]
        ind = np.random.randint(user_neg_items.shape[1], size=(batch_size, num_neg))
        neg_samples = neg_samples[batch_index[:, None], ind]

        # user_neg_neighbors = user_neg_users[batch_user_id]
        # ind = np.random.randint(user_neg_users.shape[1], size=(batch_size, num_first_order_neighbor))
        # user_neg_neighbors = user_neg_neighbors[batch_index[:, None], ind]
        #
        # item_neg_neighbors = item_neg_items[batch_item_id]
        # ind = np.random.randint(item_neg_items.shape[1], size=(batch_size, num_first_order_neighbor))
        # item_neg_neighbors = item_neg_neighbors[batch_index[:, None], ind]
        #
        # user_pos_neighbors = user_pos_users[batch_user_id]
        # ind = np.random.randint(user_pos_users.shape[1], size=(batch_size, num_first_order_neighbor))
        # user_pos_neighbors = user_pos_neighbors[batch_index[:, None], ind]
        #
        # item_pos_neighbors = item_pos_items[batch_item_id]
        # ind = np.random.randint(item_pos_items.shape[1], size=(batch_size, num_first_order_neighbor))
        # item_pos_neighbors = item_pos_neighbors[batch_index[:, None], ind]

        return batch_user_id, batch_item_id, neg_samples

    # np.random.seed(SEED)
    user_item_pairs = np.asarray(train_matrix.todok().nonzero()).T
    pair_ids = np.arange(len(user_item_pairs))
    # num_first_order_neighbor = 5
    user_neg_items = pre_samples['user_neg_items']
    # user_neg_users = pre_samples['user_neg_users']
    # item_neg_items = pre_samples['item_neg_items']
    # user_pos_users = pre_samples['user_pos_users']
    # item_pos_items = pre_samples['item_pos_items']
    while True:
        batch_data = sample()
        result_queue.put(batch_data)


class NegSampler(object):
    def __init__(self, train_matrix, pre_samples, batch_size=512, num_neg=3, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 20)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(train_matrix,
                                                      pre_samples,
                                                      batch_size,
                                                      num_neg,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()