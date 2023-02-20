import torch
from torch import nn
import torch.nn.functional as F
from utils.math_utils import arcosh, cosh, sinh
from torch.nn import Parameter
from manifolds.hyperboloid import Hyperboloid
import manifolds


class Model(nn.Module):
    def __init__(self, num_users, num_items, hidden_dim, device):
        super(Model, self).__init__()
        self.manifold = Hyperboloid()
        self.eps = {torch.float32: 1e-7, torch.float64: 1e-15}
        self.min_norm = 1e-15
        self.max_norm = 1e6

        self.num_users = num_users
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.device = device
        self.curvatures = 1.0
        self.c = torch.tensor([1.0]).to(device)

        self.user_eu_embeddings = torch.zeros(num_users, hidden_dim).type(torch.FloatTensor).to(self.device)
        self.item_eu_embeddings = torch.zeros(num_items, hidden_dim).type(torch.FloatTensor).to(self.device)

        self.user_eu_embeddings = torch.nn.init.uniform_(self.user_eu_embeddings, -0.1, 0.1)
        self.item_eu_embeddings = torch.nn.init.uniform_(self.item_eu_embeddings, -0.1, 0.1)

        self.user_hb_embeddings = self.manifold.expmap0(u=self.user_eu_embeddings, c=self.c)
        self.item_hb_embeddings = self.manifold.expmap0(u=self.item_eu_embeddings, c=self.c)

        self.user_hb_embeddings.requires_grad = True
        self.item_hb_embeddings.requires_grad = True

        # self.user_hb_embeddings = manifolds.ManifoldParameter(self.user_hb_embeddings, True, self.manifold, self.c)
        # self.item_hb_embeddings = manifolds.ManifoldParameter(self.item_hb_embeddings, True, self.manifold, self.c)

        self.myparameters = [self.user_hb_embeddings, self.item_hb_embeddings]

    def forward(self, user_emb, pos_emb, neg_emb):
        user_hyp = self.manifold.proj(user_emb, c=self.c)
        pos_hyp = self.manifold.proj(pos_emb, c=self.c)
        neg_hyp = self.manifold.proj(neg_emb, c=self.c)
        return user_hyp, pos_hyp, neg_hyp

    # def project_to_hyperbolic(self, euclidean_emb, curvature):
    #     o_item = torch.zeros_like(euclidean_emb).to(euclidean_emb.device)
    #     item_eu_embeddings0 = torch.cat([o_item[:, 0:1], euclidean_emb], dim=1)
    #     hyperbolic_emb = self.proj(self.expmap0(self.proj_tan0(
    #         item_eu_embeddings0, curvature), c=curvature), c=curvature)
    #     return hyperbolic_emb
    #
    #
    def project_to_hyperbolic(self, euclidean_emb, curvature):
        o_item = torch.zeros_like(euclidean_emb).to(euclidean_emb.device)
        item_eu_embeddings0 = torch.cat([o_item[:, 0:1], euclidean_emb], dim=1)
        # print("debug before", item_eu_embeddings0)
        # item_eu_embeddings00 = self.proj_tan0(item_eu_embeddings0, curvature)
        # print("debug after", item_eu_embeddings0)
        # print("debug", item_eu_embeddings0.sum()==item_eu_embeddings00.sum())
        hyperbolic_emb = self.expmap0(item_eu_embeddings0, c=curvature)
        return hyperbolic_emb

    def predict(self, batch_user_ind):
        batch_user_ind = torch.from_numpy(batch_user_ind).type(torch.LongTensor).to(self.device)
        with torch.no_grad():
            user_emb_v = self.user_hb_embeddings[batch_user_ind].view(-1, self.hidden_dim)
            item_emb_v = self.item_hb_embeddings.view(-1, self.hidden_dim)
            user_emb_hyper = self.manifold.proj(user_emb_v, self.c)  # b * (d+1)
            item_emb_hyper = self.manifold.proj(item_emb_v, self.c)  # item_num * (d+1)
            # print("user_emb_hyper", user_emb_hyper.shape)
            # print("item_emb_hyper", item_emb_hyper.shape)
            distance = self.hb_pairwise_distances(user_emb_hyper, item_emb_hyper, self.c)
        # self.sqdist(user_emb_hyper, pos_emb_hyper, 1)
        # batchsize * 24634
        return distance

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def cal_wass_dist(self, user_mu, user_var, item_mu, item_var):
        mu_dist = torch.pow(user_mu - item_mu, 2).sum(1).unsqueeze(1)
        sigma_dist = torch.pow(user_var - item_var, 2).sum(1).unsqueeze(1)

        return mu_dist + sigma_dist

    def KL_loss(self, mu, logvar):
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return KLD

    def pairwise_mu_distances(self, x, y=None):
        """
        reference: https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2
        :param x: x is a Nxd matrix
        :param y: y is an optional Mxd matrix
        :return: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'.
                i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        """
        x_norm = (x ** 2).sum(1).view(-1, 1)
        if y is not None:
            y_t = torch.transpose(y, 0, 1)
            y_norm = (y ** 2).sum(1).view(1, -1)
        else:
            y_t = torch.transpose(x, 0, 1)
            y_norm = x_norm.view(1, -1)

        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)

        # replace nan values with 0
        dist[dist != dist] = 0
        return dist

    def pairwise_sigma_distances(self, x, y=None):
        x_norm = (x ** 2).sum(1).view(-1, 1)
        if y is not None:
            y_t = torch.transpose(y, 0, 1)
            y_norm = (y ** 2).sum(1).view(1, -1)
        else:
            y_t = torch.transpose(x, 0, 1)
            y_norm = x_norm.view(1, -1)

        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)

        # replace nan values with 0
        dist[dist != dist] = 0
        return dist

    def minkowski_dot(self, x, y, keepdim=True):
        res = torch.sum(x * y, dim=-1) - 2 * x[..., 0] * y[..., 0]
        if keepdim:
            res = res.view(res.shape + (1,))
        return res

    def sqdist(self, x, y, c):
        K = 1. / c
        prod = self.minkowski_dot(x, y)
        theta = torch.clamp(-prod / K, min=1.0 + self.eps[x.dtype])
        sqdist = K * arcosh(theta) ** 2
        # clamp distance to avoid nans in Fermi-Dirac decoder
        return torch.clamp(sqdist, max=50.0)

    def hb_pairwise_distances(self, x, y, c):
        '''
        :param x: user: B * dim
        :param y: all_item: item_num * dim
        :return:
        '''
        dist = torch.zeros(y.shape[0], 1).to(self.device)
        for i in range(x.shape[0]):
            # if i % 1000 == 0:
            #     print(i)
            cur_x = x[i:i+1].repeat(y.shape[0], 1)
            cur_dist = self.sqdist(cur_x, y, c)
            # print("cur_dist", cur_dist.shape)
            dist = torch.cat((dist, cur_dist), 1)
        dist = dist.permute(1, 0)
        # print("dist", dist.shape)
        return dist[1:]


    def pairwise_distances(self, x, y=None):
        """
        reference: https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2
        :param x: x is a Nxd matrix
        :param y: y is an optional Mxd matrix
        :return: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'.
                i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        """
        x_norm = (x ** 2).sum(1).view(-1, 1)
        if y is not None:
            y_t = torch.transpose(y, 0, 1)
            y_norm = (y ** 2).sum(1).view(1, -1)
        else:
            y_t = torch.transpose(x, 0, 1)
            y_norm = x_norm.view(1, -1)

        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)

        # replace nan values with 0
        dist[dist != dist] = 0
        return dist

    def expmap0(self, u, c):
        K = 1. / c
        sqrtK = K ** 0.5
        d = u.size(-1) - 1
        x = u.narrow(-1, 1, d).view(-1, d)
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x_norm = torch.clamp(x_norm, min=self.min_norm)
        theta = x_norm / sqrtK
        res = torch.ones_like(u)
        res[:, 0:1] = sqrtK * cosh(theta)
        res[:, 1:] = sqrtK * sinh(theta) * x / x_norm
        return self.proj(res, c)


class Controller(nn.Module):
    def __init__(self, hidden_dim, device):
        super(Controller, self).__init__()

        self.linear1 = nn.Linear(hidden_dim, 20, bias=True).to(device)
        self.linear2 = nn.Linear(20, 1, bias=True).to(device)

    def forward(self, x):
        z = self.linear1(x)
        margin = F.softplus(self.linear2(z))
        return margin


class ManifoldParameter(Parameter):
    """
    Subclass of torch.nn.Parameter for Riemannian optimization.
    """
    def __new__(cls, data, requires_grad, manifold, c):
        return Parameter.__new__(cls, data, requires_grad)

    def __init__(self, data, requires_grad, manifold, c):
        self.c = c
        self.manifold = manifold

    def __repr__(self):
        return '{} Parameter containing:\n'.format(self.manifold.name) + super(Parameter, self).__repr__()