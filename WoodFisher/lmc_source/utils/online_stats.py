"""
Online-ish Pearson correlation of all n x n variable pairs simultaneously.
Adapt from git re-basin repo:
https://github.com/samuela/git-re-basin/blob/main/src/online_stats.py.
Modify the original code from jax to pytorch.
"""
import torch


class OnlineMean:
    def __init__(self, num_features, device=None):
        self.sum = torch.zeros(num_features).to(device)
        self.count = 0

    def update(self, batch):
        # batch shape: (batch_size, channels, width, height)
        # or
        # batch shape: (batch_size, num_features)
        if len(batch.shape) == 4:
            self.sum += torch.sum(batch, dim=(0, 2, 3))
        elif len(batch.shape) == 2:
            self.sum += torch.sum(batch, dim=0)
        else:
            raise ValueError("batch shape must be (batch_size, channels, "
                             "width, height) or (batch_size, num_features)")
        self.count += batch.shape[0]

    def mean(self):
        return self.sum / self.count


class OnlineCovariance:
    def __init__(self, a_mean, b_mean, count, device=None):
        assert a_mean.shape == b_mean.shape
        d = a_mean.shape[0]
        # ensure brodcast calculation
        self.a_mean = torch.zeros_like(a_mean).double()
        self.b_mean = torch.zeros_like(b_mean).double()
        self.cov = torch.zeros((d, d)).to(device).double()
        self.std_a = torch.zeros_like(a_mean).double()
        self.std_b = torch.zeros_like(b_mean).double()
        self.count = count

    def update(self, a_batch, b_batch):
        assert a_batch.shape == b_batch.shape
        a_batch = a_batch.double()
        b_batch = b_batch.double()
        self.a_mean += a_batch.mean(dim=1) / self.count
        self.b_mean += b_batch.mean(dim=1) / self.count
        self.cov += torch.matmul(a_batch, b_batch.T) / a_batch.shape[1] / self.count
        self.std_a += a_batch.std(dim=1) / self.count
        self.std_b += b_batch.std(dim=1) / self.count

    def pearson_correlation(self):
        eps = 1e-4

        self.cov = self.cov - torch.outer(self.a_mean, self.b_mean)
        return torch.nan_to_num(self.cov / (torch.outer(self.std_a, self.std_b) + eps))


class OnlineCovariance_Git:
    def __init__(self, a_mean, b_mean, device=None):
        assert a_mean.shape == b_mean.shape
        d = a_mean.shape[0]
        # ensure brodcast calculation
        self.a_mean = a_mean.reshape(-1, 1)
        self.b_mean = b_mean.reshape(-1, 1)
        self.cov = torch.zeros((d, d)).to(device)
        self.var_a = torch.zeros(d).to(device)
        self.var_b = torch.zeros(d).to(device)
        self.count = 0

    def update(self, a_batch, b_batch):
        assert a_batch.shape == b_batch.shape
        batch_size, _ = a_batch.shape
        a_res = a_batch - self.a_mean
        b_res = b_batch - self.b_mean
        self.cov += torch.matmul(a_res, b_res.T)
        self.var_a += torch.sum(a_res**2, dim=1)
        self.var_b += torch.sum(b_res**2, dim=1)
        self.count += batch_size

    def covariance(self):
        return self.cov / (self.count - 1)

    def a_variance(self):
        return self.var_a / (self.count - 1)

    def b_variance(self):
        return self.var_b / (self.count - 1)

    def a_stddev(self):
        return torch.sqrt(self.a_variance())

    def b_stddev(self):
        return torch.sqrt(self.b_variance())

    def E_ab(self):
        return self.covariance() + torch.outer(self.a_mean, self.b_mean)

    def pearson_correlation(self):
        eps = 0
        return torch.nan_to_num(self.cov / (torch.sqrt(self.var_a[:, None]) +
                                            eps) /
                                (torch.sqrt(self.var_b) + eps))
