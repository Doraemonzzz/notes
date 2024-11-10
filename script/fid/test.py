import numpy as np
import torch
from scipy import linalg
from torch import Tensor

def _compute_fid(mu1: Tensor, sigma1: Tensor, mu2: Tensor, sigma2: Tensor) -> Tensor:
    r"""Compute adjusted version of `Fid Score`_.

    The Frechet Inception Distance between two multivariate Gaussians X_x ~ N(mu_1, sigm_1)
    and X_y ~ N(mu_2, sigm_2) is d^2 = ||mu_1 - mu_2||^2 + Tr(sigm_1 + sigm_2 - 2*sqrt(sigm_1*sigm_2)).

    Args:
        mu1: mean of activations calculated on predicted (x) samples
        sigma1: covariance matrix over activations calculated on predicted (x) samples
        mu2: mean of activations calculated on target (y) samples
        sigma2: covariance matrix over activations calculated on target (y) samples

    Returns:
        Scalar value of the distance between sets.

    """
    a = (mu1 - mu2).square().sum(dim=-1)
    b = sigma1.trace() + sigma2.trace()
    c = torch.linalg.eigvals(sigma1 @ sigma2).sqrt().real.sum(dim=-1)

    return a + b - 2 * c

b = 50000
d = 2048

x1 = torch.randn(b, d)
x2 = torch.randn(b, d)

##### torchmetric
mu1 = torch.mean(x1, dim=0, keepdim=True)
mu2 = torch.mean(x2, dim=0, keepdim=True)

cov_sum1 = x1.t().mm(x1)
cov_sum2 = x2.t().mm(x2)
cov1 = (cov_sum1 - b * mu1.t().mm(mu1)) / (b - 1)
cov2 = (cov_sum2 - b * mu2.t().mm(mu2)) / (b - 1)

fid1 = _compute_fid(mu1.squeeze(0), cov1, mu2.squeeze(0), cov2)


##### openai
def get_mu_sigma(x):
    x = x.cpu().numpy()
    mu = np.mean(x, axis=0)
    sigma = np.cov(x, rowvar=False)
    
    return mu, sigma

def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Compute the Frechet distance between two sets of statistics.
    """
    # https://github.com/bioinf-jku/TTUR/blob/73ab375cdf952a12686d9aa7978567771084da42/fid.py#L132
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), f"Training and test mean vectors have different lengths: {mu1.shape}, {mu2.shape}"
    assert (
        sigma1.shape == sigma2.shape
    ), f"Training and test covariances have different dimensions: {sigma1.shape}, {sigma2.shape}"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; adding %s to diagonal of cov estimates"
            % eps
        )
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

mu1, sigma1 = get_mu_sigma(x1)
mu2, sigma2 = get_mu_sigma(x2)

fid2 = frechet_distance(mu1, sigma1, mu2, sigma2)


print(np.linalg.norm(fid1 - fid2))