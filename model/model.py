import torch
from torch import nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_users, num_items, hidden_dim, device):
        super(Model, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.device = device


        self.user_mu_embeddings = torch.zeros(num_users, hidden_dim).type(torch.FloatTensor).to(self.device)
        self.item_mu_embeddings = torch.zeros(num_items, hidden_dim).type(torch.FloatTensor).to(self.device)
        self.user_mu_embeddings.requires_grad = True
        self.item_mu_embeddings.requires_grad = True

        self.user_mu_embeddings = torch.nn.init.normal_(self.user_mu_embeddings, 0, 0.01)
        self.item_mu_embeddings = torch.nn.init.normal_(self.item_mu_embeddings, 0, 0.01)


        self.myparameters = [self.user_mu_embeddings, self.item_mu_embeddings]


    def forward(self, user_emb, pos_emb, neg_emb):
        pos_dist = torch.pow(user_emb - pos_emb, 2).sum(1).unsqueeze(1)
        neg_dist = torch.pow(user_emb - neg_emb, 2).sum(1).unsqueeze(1)

        return pos_dist, neg_dist

    def predict(self, batch_user_ind):
        batch_user_ind = torch.from_numpy(batch_user_ind).type(torch.LongTensor).to(self.device)
        user_mu = self.user_mu_embeddings[batch_user_ind]
        distance = self.pairwise_distances(user_mu, self.item_mu_embeddings)
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


class Controller(nn.Module):
    def __init__(self, hidden_dim, device):
        super(Controller, self).__init__()

        self.linear1 = nn.Linear(hidden_dim, 20, bias=True).to(device)
        self.linear2 = nn.Linear(20, 1, bias=True).to(device)

    def forward(self, x):
        z = self.linear1(x)
        margin = F.softplus(self.linear2(z))
        return margin
