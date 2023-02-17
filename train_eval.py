import sys
import datetime
import time
import logging
import numpy as np
from argparse import ArgumentParser

import torch
import torch.backends.cudnn as cudnn
from torch import autograd

from eval_metrics import precision_at_k, recall_at_k, mapk, ndcg_k
from sampler import NegSampler
from model.model import Model, Controller

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def parse_args():
    parser = ArgumentParser(description="Wass Rec")
    parser.add_argument('-margin', '--margin_rate', type=float, default=1.0, help='margin rate')
    parser.add_argument('-e', '--epoch', type=int, default=1001, help='number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=5000, help='batch size for training')
    parser.add_argument('-dim', '--hidden_dim', type=int, default=50, help='the number of the hidden dimension')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-3, help='weight decay')
    parser.add_argument('-n_neg', '--neg_samples', type=int, default=10, help='the number of negative samples')
    parser.add_argument('-dr', '--dropout_rate', type=float, default=0.5, help='the dropout probability')
    parser.add_argument('--ac_fc', type=str, default='tanh')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('-seed', type=int, default=0, help='random state to split the data')
    parser.add_argument('--data', type=str, default='cds')
    parser.add_argument('--is_logging', type=bool, default=True)
    return parser.parse_args()


def negsamp_vectorized_bsearch_preverif(pos_inds, n_items, n_samp=32):
    """ Pre-verified with binary search
    `pos_inds` is assumed to be ordered
    reference: https://tech.hbc.com/2018-03-23-negative-sampling-in-numpy.html
    """
    raw_samp = np.random.randint(0, n_items - len(pos_inds), size=n_samp)
    pos_inds_adj = pos_inds - np.arange(len(pos_inds))
    neg_inds = raw_samp + np.searchsorted(pos_inds_adj, raw_samp, side='right')
    return neg_inds

def set_seed(seed):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

class Recommender(object):
    def __init__(self, data_set, config):
        assert data_set is not None, 'The data set is not valid.'
        set_seed(config.seed)
        train_set, train_matrix, test_set = data_set.generate_dataset(config.seed)
        self.train_set = train_set
        self.train_matrix = train_matrix
        self.test_set = test_set
        self.config = config
        self.user_id_shift = 0
        self.item_id_shift = 0
        num_users, num_items = train_matrix.shape
        self.model = Model(num_users, num_items, config.hidden_dim, config.device).to(config.device)
        print("margin", self.config.margin)

    def neg_item_pre_sampling(self, train_matrix, num_neg_candidates=500):
        num_users, num_items = train_matrix.shape
        user_neg_items = []
        for user_id in range(num_users):
            pos_items = train_matrix[user_id].indices
            u_neg_item = negsamp_vectorized_bsearch_preverif(pos_items, num_items, num_neg_candidates)
            user_neg_items.append(u_neg_item)
        user_neg_items = np.asarray(user_neg_items)
        return user_neg_items

    def margin_ranking_loss(self, pos, neg, margin=1):
        device = self.config.device
        res = pos - neg + margin
        res = torch.where(res < torch.FloatTensor([0]).to(device), torch.FloatTensor([0]).to(device), res)
        return torch.sum(res)

    def bpr_loss(self, pos_rating_vector, neg_rating_vector, margin=0):
        loss = pos_rating_vector - neg_rating_vector + margin
        loss = torch.sigmoid(loss)
        loss = -torch.log(loss)
        loss = torch.sum(loss)
        return loss

    def max_norm(self, param, max_val=1, eps=1e-8):
        norm = param.norm(2, dim=1, keepdim=True)
        desired = torch.clamp(norm, 0, max_val)
        param = param * (desired / (eps + norm))
        return param

    def batch_mask(self, ids):
        unique_id = list(set(ids))
        reverse_mapping = {}
        for in_id, out_id in enumerate(unique_id):
            reverse_mapping[out_id] = in_id
        mask = np.zeros((len(unique_id), len(ids))).astype(np.float32)
        for record_id, true_id in enumerate(ids):
            mask[reverse_mapping[true_id], record_id] = 1
        return mask

    def adam_proxy(self, M, R, ind, grad, iter):
        beta1 = .9
        beta2 = .999
        eps = 1e-8
        M[ind] = beta1 * M[ind] + (1. - beta1) * grad
        R[ind] = beta2 * R[ind] + (1. - beta2) * grad ** 2
        m_k_hat = M[ind] / (1. - beta1 ** (iter))
        r_k_hat = R[ind] / (1. - beta2 ** (iter))
        updated_grad = m_k_hat / (torch.sqrt(r_k_hat) + eps)
        return updated_grad

    def train(self):
        # pre-sample a small set of negative samples
        t1 = time.time()
        user_neg_items = self.neg_item_pre_sampling(self.train_matrix, num_neg_candidates=500)
        pre_samples = {'user_neg_items': user_neg_items}
        print("Pre sampling time:{}".format(time.time() - t1))

        batch_size = self.config.batch_size
        n_neg = self.config.neg_samples
        sampler = NegSampler(self.train_matrix, pre_samples, batch_size=batch_size, num_neg=n_neg, n_workers=4)

        lr, wd = self.config.learning_rate, self.config.weight_decay
        model_optimizer = torch.optim.Adam(self.model.myparameters, lr=lr, weight_decay=wd)

        num_pairs = self.train_matrix.count_nonzero()
        num_batches = int(num_pairs / batch_size) + 1

        # M_user = torch.zeros_like(self.model.user_mu_embeddings).type(torch.FloatTensor).to(self.config.device)
        # R_user = torch.zeros_like(self.model.user_mu_embeddings).type(torch.FloatTensor).to(self.config.device)
        # M_user.requires_grad = False
        # R_user.requires_grad = False
        #
        # M_item = torch.zeros_like(self.model.item_mu_embeddings).type(torch.FloatTensor).to(self.config.device)
        # R_item = torch.zeros_like(self.model.item_mu_embeddings).type(torch.FloatTensor).to(self.config.device)
        # M_item.requires_grad = False
        # R_item.requires_grad = False
        print("begin training")
        self.model.train()
        try:
            for t in range(self.config.epoch):
                logger.debug("epoch:{}".format(t))
                avg_cost = 0.
                neg_time = 0
                t1 = time.time()

                for batchID in range(num_batches):
                    #data preparation
                    t_neg1 = time.time()
                    batch_user_id, batch_item_id, neg_samples = sampler.next_batch()
                    t_neg2 = time.time()

                    user_id = torch.from_numpy(batch_user_id).type(torch.LongTensor).to(self.config.device)
                    pos_id = torch.from_numpy(batch_item_id).type(torch.LongTensor).to(self.config.device)
                    neg_id = torch.from_numpy(neg_samples).type(torch.LongTensor).to(self.config.device)
                    user_id = user_id.unsqueeze(1).repeat(1, neg_samples.shape[1])
                    pos_id = pos_id.unsqueeze(1).repeat(1, neg_samples.shape[1])

                    user_emb = self.model.user_mu_embeddings[user_id]
                    pos_emb = self.model.item_mu_embeddings[pos_id]
                    neg_emb = self.model.item_mu_embeddings[neg_id]
                    user_emb_v = user_emb.view(-1, self.config.hidden_dim)
                    pos_emb_v = pos_emb.view(-1, self.config.hidden_dim)
                    neg_emb_v = neg_emb.view(-1, self.config.hidden_dim)
                    # print("user", user_emb_v[0:2])
                    # print("item", pos_emb_v[0:2])

                    # Rui = torch.sum(user_emb_v*pos_emb_v, dim=1)
                    # Ruj = torch.sum(user_emb_v*neg_emb_v, dim=1)
                    # loss = self.bpr_loss(Rui, Ruj)
                    Rui = torch.sum(user_emb_v**2, dim=1) + torch.sum(pos_emb_v**2, dim=1) -\
                          2.0*torch.sum(user_emb_v*pos_emb_v, dim=1)
                    Ruj = torch.sum(user_emb_v**2, dim=1) + torch.sum(neg_emb_v**2, dim=1) -\
                          2.0*torch.sum(user_emb_v*neg_emb_v, dim=1)

                    loss = self.margin_ranking_loss(Rui, Ruj, margin=self.config.margin)
                    model_optimizer.zero_grad()
                    loss.backward()
                    model_optimizer.step()

                    avg_cost += loss / num_batches
                    neg_time += t_neg2 - t_neg1

                logger.debug("Avg loss:{}".format(avg_cost))
                print("neg time is {}".format(neg_time))
                print("all time is {}".format(time.time() - t1))
                if t % 100 == 0 and t > 0:
                    sampler.close()
                    user_neg_items = self.neg_item_pre_sampling(self.train_matrix, num_neg_candidates=500)
                    pre_samples = {'user_neg_items': user_neg_items}
                    sampler = NegSampler(self.train_matrix, pre_samples, batch_size=batch_size,
                                         num_neg=n_neg, n_workers=4)
                    t2 = time.time()
                    precision, recall, MAP, ndcg = self.evaluate(1024)
                    print("recall10 is {} sota recall10 is 0.073 \n ndcg10 is {} sota ndcg10 is 0.0383".format(recall[1], ndcg[1]))
                    logger.info("Evaluation time:{}".format(time.time() - t2))
            sampler.close()
        except KeyboardInterrupt:
            sampler.close()
            sys.exit()

    def evaluate(self, num_batch_users=1024):
        num_users, num_items = self.train_matrix.shape
        num_batches = int(num_users / num_batch_users) + 1
        user_indexes = np.arange(self.user_id_shift, num_users)
        topk = 50
        precision, recall, MAP, ndcg = [], [], [], []
        pred_list = None

        for batchID in range(num_batches):
            start = batchID * num_batch_users
            end = start + num_batch_users

            if batchID == num_batches - 1:
                if start < num_users:
                    end = num_users
                else:
                    break

            batch_user_index = user_indexes[start:end]

            pred_distance = self.model.predict(batch_user_index)
            # print("pred_distance", pred_distance.shape)
            pred_distance = pred_distance.cpu().data.numpy().copy()

            # eliminate the training items in the prediction list
            pred_distance[self.train_matrix[batch_user_index].nonzero()] = np.inf
            pred_distance[:, 0] = np.inf
            ind = np.argpartition(pred_distance, topk)
            ind = ind[:, :topk]
            arr_ind = pred_distance[np.arange(len(pred_distance))[:, None], ind]
            arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(pred_distance))]
            pred_items = ind[np.arange(len(pred_distance))[:, None], arr_ind_argsort]

            if batchID == 0:
                pred_list = pred_items.copy()
            else:
                pred_list = np.append(pred_list, pred_items, axis=0)

        for k in [5, 10, 15, 20]:
            precision.append(precision_at_k(self.test_set, pred_list, k))
            recall.append(recall_at_k(self.test_set, pred_list, k))
            MAP.append(mapk(self.test_set, pred_list, k))
            ndcg.append(ndcg_k(self.test_set, pred_list, k))

        logger.info(', '.join(str(e) for e in precision))
        logger.info(', '.join(str(e) for e in recall))
        logger.info(', '.join(str(e) for e in MAP))
        logger.info(', '.join(str(e) for e in ndcg))

        return precision, recall, MAP, ndcg

    def run(self):
        self.train()
        self.evaluate()

        logger.info('Parameters:')
        for arg, value in sorted(vars(self.config).items()):
            logger.info("%s: %r", arg, value)
        logger.info('\n')


def main():
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    config = parse_args()

    #for reproduction
    set_seed(config.seed)

    # config the gpu device
    config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    print("config.device", config.device)

    # config if write the log to a file
    if config.is_logging is True:
        handler = logging.FileHandler('./log/final/' + config.data + '.log')
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)

    # read the data
    from data import Amazon, MovieLens, GoodReads
    if config.data.lower() == 'cds':
        data_set = Amazon.CDs()
    elif config.data.lower() == 'books':
        data_set = Amazon.Books()
    elif config.data.lower() == 'children':
        data_set = GoodReads.Children()
    elif config.data.lower() == 'comics':
        data_set = GoodReads.Comics()
    elif config.data.lower() == 'ml20m':
        data_set = MovieLens.ML20M()
    else:
        data_set = None
    print("'data load over")
    rec = Recommender(data_set, config)
    rec.run()


if __name__ == '__main__':
    main()
