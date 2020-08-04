from typing import Sequence

import torch
from torch.nn import functional as F

from pytorch_lightning.metrics.functional.reduction import reduce


def mse(
        pred: torch.Tensor,
        target: torch.Tensor,
        reduction: str = 'elementwise_mean'
) -> torch.Tensor:
    """
    Computes mean squared error

    Args:
        pred: estimated labels
        target: ground truth labels
        reduction: method for reducing mse (default: takes the mean)
            Available reduction methods:

            - elementwise_mean: takes the mean
            - none: pass array
            - sum: add elements

    Return:
        Tensor with MSE

    Example:

        >>> x = torch.tensor([0., 1, 2, 3])
        >>> y = torch.tensor([0., 1, 2, 2])
        >>> mse(x, y)
        tensor(0.2500)

    """
    mse = F.mse_loss(pred, target, reduction='none')
    mse = reduce(mse, reduction=reduction)
    return mse


def rmse(
        pred: torch.Tensor,
        target: torch.Tensor,
        reduction: str = 'elementwise_mean'
) -> torch.Tensor:
    """
    Computes root mean squared error

    Args:
        pred: estimated labels
        target: ground truth labels
        reduction: method for reducing rmse (default: takes the mean)
            Available reduction methods:

            - elementwise_mean: takes the mean
            - none: pass array
            - sum: add elements

    Return:
        Tensor with RMSE


        >>> x = torch.tensor([0., 1, 2, 3])
        >>> y = torch.tensor([0., 1, 2, 2])
        >>> rmse(x, y)
        tensor(0.5000)

    """
    rmse = torch.sqrt(mse(pred, target, reduction=reduction))
    return rmse


def mae(
        pred: torch.Tensor,
        target: torch.Tensor,
        reduction: str = 'elementwise_mean'
) -> torch.Tensor:
    """
    Computes mean absolute error

    Args:
        pred: estimated labels
        target: ground truth labels
        reduction: method for reducing mae (default: takes the mean)
            Available reduction methods:

            - elementwise_mean: takes the mean
            - none: pass array
            - sum: add elements

    Return:
        Tensor with MAE

    Example:

        >>> x = torch.tensor([0., 1, 2, 3])
        >>> y = torch.tensor([0., 1, 2, 2])
        >>> mae(x, y)
        tensor(0.2500)

    """
    mae = F.l1_loss(pred, target, reduction='none')
    mae = reduce(mae, reduction=reduction)
    return mae


def rmsle(
        pred: torch.Tensor,
        target: torch.Tensor,
        reduction: str = 'elementwise_mean'
) -> torch.Tensor:
    """
    Computes root mean squared log error

    Args:
        pred: estimated labels
        target: ground truth labels
        reduction: method for reducing rmsle (default: takes the mean)
            Available reduction methods:

            - elementwise_mean: takes the mean
            - none: pass array
            - sum: add elements

    Return:
        Tensor with RMSLE

    Example:

        >>> x = torch.tensor([0., 1, 2, 3])
        >>> y = torch.tensor([0., 1, 2, 2])
        >>> rmsle(x, y)
        tensor(0.0207)

    """
    rmsle = mse(torch.log(pred + 1), torch.log(target + 1), reduction=reduction)
    return rmsle


def psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = None,
    base: float = 10.0,
    reduction: str = 'elementwise_mean'
) -> torch.Tensor:
    """
    Computes the peak signal-to-noise ratio

    Args:
        pred: estimated signal
        target: groun truth signal
        data_range: the range of the data. If None, it is determined from the data (max - min)
        base: a base of a logarithm to use (default: 10)
        reduction: method for reducing psnr (default: takes the mean)
            Available reduction methods:

            - elementwise_mean: takes the mean
            - none: pass array
            - sum add elements

    Return:
        Tensor with PSNR score

    Example:

        >>> from pytorch_lightning.metrics.regression import PSNR
        >>> pred = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
        >>> target = torch.tensor([[3.0, 2.0], [1.0, 0.0]])
        >>> metric = PSNR()
        >>> metric(pred, target)
        tensor(2.5527)

    """

    if data_range is None:
        data_range = max(target.max() - target.min(), pred.max() - pred.min())
    else:
        data_range = torch.tensor(float(data_range))

    mse_score = mse(pred.view(-1), target.view(-1), reduction=reduction)
    psnr_base_e = 2 * torch.log(data_range) - torch.log(mse_score)
    psnr = psnr_base_e * (10 / torch.log(torch.tensor(base)))
    return psnr


def _gaussian_kernel(channel, kernel_size, sigma, device):
    def gaussian(kernel_size, sigma, device):
        gauss = torch.arange(
            start=(1 - kernel_size) / 2, end=(1 + kernel_size) / 2, step=1, dtype=torch.float32, device=device
        )
        gauss = torch.exp(-gauss.pow(2) / (2 * pow(sigma, 2)))
        return (gauss / gauss.sum()).unsqueeze(dim=0)  # (1, kernel_size)

    gaussian_kernel_x = gaussian(kernel_size[0], sigma[0], device)
    gaussian_kernel_y = gaussian(kernel_size[1], sigma[1], device)
    kernel = torch.matmul(gaussian_kernel_x.t(), gaussian_kernel_y)  # (kernel_size, 1) * (1, kernel_size)

    return kernel.expand(channel, 1, kernel_size[0], kernel_size[1])


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    kernel_size: Sequence[int] = (11, 11),
    sigma: Sequence[float] = (1.5, 1.5),
    reduction: str = "elementwise_mean",
    data_range: float = None,
    k1: float = 0.01,
    k2: float = 0.03
) -> torch.Tensor:
    """
    Computes Structual Similarity Index Measure

    Args:
        pred: Estimated image
        target: Ground truth image
        kernel_size: Size of the gaussian kernel. Default: (11, 11)
        sigma: Standard deviation of the gaussian kernel. Default: (1.5, 1.5)
        reduction: A method for reducing ssim over all elements in the ``pred`` tensor. Default: ``elementwise_mean``

            Available reduction methods:
            - elementwise_mean: takes the mean
            - none: pass away
            - sum: add elements

        data_range: Range of the image. If ``None``, it is determined from the image (max - min)
        k1: Parameter of SSIM. Default: 0.01
        k2: Parameter of SSIM. Default: 0.03

    Returns:
        A Tensor with SSIM

    Example:

        >>> pred = torch.rand([16, 1, 16, 16])
        >>> target = pred * 1.25
        >>> ssim(pred, target)
        tensor(0.9520)
    """

    if pred.dtype != target.dtype:
        raise TypeError(
            "Expected `pred` and `target` to have the same data type."
            f" Got pred: {pred.dtype} and target: {target.dtype}."
        )

    if pred.shape != target.shape:
        raise ValueError(
            "Expected `pred` and `target` to have the same shape."
            f" Got pred: {pred.shape} and target: {target.shape}."
        )

    if len(pred.shape) != 4 or len(target.shape) != 4:
        raise ValueError(
            "Expected `pred` and `target` to have BxCxHxW shape."
            f" Got pred: {pred.shape} and target: {target.shape}."
        )

    if len(kernel_size) != 2 or len(sigma) != 2:
        raise ValueError(
            "Expected `kernel_size` and `sigma` to have the length of two."
            f" Got kernel_size: {len(kernel_size)} and sigma: {len(sigma)}."
        )

    if any(x % 2 == 0 or x <= 0 for x in kernel_size):
        raise ValueError(f"Expected `kernel_size` to have odd positive number. Got {kernel_size}.")

    if any(y <= 0 for y in sigma):
        raise ValueError(f"Expected `sigma` to have positive number. Got {sigma}.")

    if data_range is None:
        data_range = max(pred.max() - pred.min(), target.max() - target.min())

    C1 = pow(k1 * data_range, 2)
    C2 = pow(k2 * data_range, 2)
    device = pred.device

    channel = pred.size(1)
    kernel = _gaussian_kernel(channel, kernel_size, sigma, device)
    mu_pred = F.conv2d(pred, kernel, groups=channel)
    mu_target = F.conv2d(target, kernel, groups=channel)

    mu_pred_sq = mu_pred.pow(2)
    mu_target_sq = mu_target.pow(2)
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = F.conv2d(pred * pred, kernel, groups=channel) - mu_pred_sq
    sigma_target_sq = F.conv2d(target * target, kernel, groups=channel) - mu_target_sq
    sigma_pred_target = F.conv2d(pred * target, kernel, groups=channel) - mu_pred_target

    UPPER = 2 * sigma_pred_target + C2
    LOWER = sigma_pred_sq + sigma_target_sq + C2

    ssim_idx = ((2 * mu_pred_target + C1) * UPPER) / ((mu_pred_sq + mu_target_sq + C1) * LOWER)

    return reduce(ssim_idx, reduction)
