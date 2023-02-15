import torch
from torch import nn
import torch.nn.functional as F
from utils.math_utils import arcosh, cosh, sinh


class Model(nn.Module):
    def __init__(self, num_users, num_items, hidden_dim, device):
        super(Model, self).__init__()
        self.eps = {torch.float32: 1e-7, torch.float64: 1e-15}
        self.min_norm = 1e-15
        self.max_norm = 1e6

        self.num_users = num_users
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.device = device
        self.curvatures = 1.0

        self.user_mu_embeddings = torch.zeros(num_users, hidden_dim).type(torch.FloatTensor).to(self.device)
        self.item_mu_embeddings = torch.zeros(num_items, hidden_dim).type(torch.FloatTensor).to(self.device)
        self.user_mu_embeddings.requires_grad = True
        self.item_mu_embeddings.requires_grad = True

        self.user_mu_embeddings = torch.nn.init.normal_(self.user_mu_embeddings, 0, 0.01)
        self.item_mu_embeddings = torch.nn.init.normal_(self.item_mu_embeddings, 0, 0.01)

        # project users and items to hyperboloid
        self.o_user = torch.zeros_like(self.user_mu_embeddings).to(self.device)
        self.o_item = torch.zeros_like(self.item_mu_embeddings).to(self.device)

        # self.user_eu_embeddings0 = torch.cat([self.o_user[:, 0:1], self.user_eu_embeddings], dim=1)
        # self.item_eu_embeddings0 = torch.cat([self.o_item[:, 0:1], self.item_eu_embeddings], dim=1)

        # self.user_hy_embeddings = self.proj(self.expmap0(self.proj_tan0(
        #     self.user_eu_embeddings0, self.curvatures), c=self.curvatures), c=self.curvatures)
        #
        # self.item_hy_embeddings = self.proj(self.expmap0(self.proj_tan0(
        #     self.item_eu_embeddings, self.curvatures), c=self.curvatures), c=self.curvatures)

        self.myparameters = [self.user_mu_embeddings, self.item_mu_embeddings]

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

    def proj_tan0(self, u, c):
        narrowed = u.narrow(-1, 0, 1)
        vals = torch.zeros_like(u)
        vals[:, 0:1] = narrowed
        return u - vals

    def proj(self, x, c):
        K = 1. / c
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d)
        y_sqnorm = torch.norm(y, p=2, dim=1, keepdim=True) ** 2
        mask = torch.ones_like(x)
        mask[:, 0] = 0
        vals = torch.zeros_like(x)
        vals[:, 0:1] = torch.sqrt(torch.clamp(K + y_sqnorm, min=self.eps[x.dtype]))
        return vals + mask * x

    def forward(self, user_emb, pos_emb, neg_emb):
        pos_dist = torch.pow(user_emb - pos_emb, 2).sum(1).unsqueeze(1)
        neg_dist = torch.pow(user_emb - neg_emb, 2).sum(1).unsqueeze(1)

        return pos_dist, neg_dist

    def project_to_hyperbolic(self, euclidean_emb, curvature):
        o_item = torch.zeros_like(euclidean_emb).to(euclidean_emb.device)
        item_eu_embeddings0 = torch.cat([o_item[:, 0:1], euclidean_emb], dim=1)
        hyperbolic_emb = self.proj(self.expmap0(self.proj_tan0(
            item_eu_embeddings0, curvature), c=curvature), c=curvature)
        return hyperbolic_emb

    def predict(self, batch_user_ind):
        batch_user_ind = torch.from_numpy(batch_user_ind).type(torch.LongTensor).to(self.device)
        user_mu = self.user_mu_embeddings[batch_user_ind]
        item_mu = self.item_mu_embeddings
        user_emb_v = user_mu.view(-1, self.hidden_dim)
        item_emb_v = item_mu.view(-1, self.hidden_dim)

        curvature = 1.0
        user_emb_hyper = self.project_to_hyperbolic(user_emb_v, curvature)  # b * (d+1)
        item_emb_hyper = self.project_to_hyperbolic(item_emb_v, curvature)  # item_num * (d+1)
        # print("user_emb_hyper", user_emb_hyper.shape)
        # print("item_emb_hyper", item_emb_hyper.shape)

        distance = self.hb_pairwise_distances(user_emb_hyper, item_emb_hyper, curvature)
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
            if i % 1000 == 0:
                print(i)
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
