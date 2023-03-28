from math import floor, ceil
from typing import List

import torch


def invert_grid(grid, input_shape, separable=False):
    f = invert_separable_grid if separable else invert_nonseparable_grid
    return f(grid, list(input_shape))


@torch.jit.script
def invert_separable_grid(grid, input_shape: List[int]):
    grid = grid.clone()
    device = grid.device
    H: int = input_shape[2]
    W: int = input_shape[3]
    B, grid_H, grid_W, _ = grid.shape
    assert B == input_shape[0]

    eps = 1e-8
    grid[:, :, :, 0] = (grid[:, :, :, 0] + 1) / 2 * (W - 1)
    grid[:, :, :, 1] = (grid[:, :, :, 1] + 1) / 2 * (H - 1)
    # grid now ranges from 0 to ([H or W] - 1)
    # TODO: implement batch operations
    inverse_grid = 2 * max(H, W) * torch.ones(
        [B, H, W, 2], dtype=torch.float32, device=device)
    for b in range(B):
        # each of these is ((grid_H - 1)*(grid_W - 1)) x 2
        p00 = grid[b,  :-1,   :-1, :].contiguous().view(-1, 2)  # noqa: 203
        p10 = grid[b, 1:  ,   :-1, :].contiguous().view(-1, 2)  # noqa: 203
        p01 = grid[b,  :-1,  1:  , :].contiguous().view(-1, 2)  # noqa: 203

        ref = torch.floor(p00).to(torch.int)
        v00 = p00 - ref
        v10 = p10 - ref
        v01 = p01 - ref
        vx = p01[:, 0] - p00[:, 0]
        vy = p10[:, 1] - p00[:, 1]

        min_x = int(floor(v00[:, 0].min() - eps))
        max_x = int(ceil(v01[:, 0].max() + eps))
        min_y = int(floor(v00[:, 1].min() - eps))
        max_y = int(ceil(v10[:, 1].max() + eps))

        pts = torch.cartesian_prod(
            torch.arange(min_x, max_x + 1, device=device),
            torch.arange(min_y, max_y + 1, device=device),
        ).T  # 2 x (x_range*y_range)

        unwarped_x = (pts[0].unsqueeze(0) - v00[:, 0].unsqueeze(1)) / vx.unsqueeze(1)  # noqa: E501
        unwarped_y = (pts[1].unsqueeze(0) - v00[:, 1].unsqueeze(1)) / vy.unsqueeze(1)  # noqa: E501
        unwarped_pts = torch.stack((unwarped_y, unwarped_x), dim=0)  # noqa: E501, has shape2 x ((grid_H - 1)*(grid_W - 1)) x (x_range*y_range)

        good_indices = torch.logical_and(
            torch.logical_and(-eps <= unwarped_pts[0],
                              unwarped_pts[0] <= 1+eps),
            torch.logical_and(-eps <= unwarped_pts[1],
                              unwarped_pts[1] <= 1+eps),
        )  # ((grid_H - 1)*(grid_W - 1)) x (x_range*y_range)
        nonzero_good_indices = good_indices.nonzero()
        inverse_j = pts[0, nonzero_good_indices[:, 1]] + ref[nonzero_good_indices[:, 0], 0]  # noqa: E501
        inverse_i = pts[1, nonzero_good_indices[:, 1]] + ref[nonzero_good_indices[:, 0], 1]  # noqa: E501
        # TODO: is replacing this with reshape operations on good_indices faster? # noqa: E501
        j = nonzero_good_indices[:, 0] % (grid_W - 1)
        i = nonzero_good_indices[:, 0] // (grid_W - 1)
        grid_mappings = torch.stack(
            (j + unwarped_pts[1, good_indices], i + unwarped_pts[0, good_indices]),  # noqa: E501
            dim=1
        )
        in_bounds = torch.logical_and(
            torch.logical_and(0 <= inverse_i, inverse_i < H),
            torch.logical_and(0 <= inverse_j, inverse_j < W),
        )
        inverse_grid[b, inverse_i[in_bounds], inverse_j[in_bounds], :] = grid_mappings[in_bounds, :]  # noqa: E501

    inverse_grid[..., 0] = (inverse_grid[..., 0]) / (grid_W - 1) * 2.0 - 1.0  # noqa: E501
    inverse_grid[..., 1] = (inverse_grid[..., 1]) / (grid_H - 1) * 2.0 - 1.0  # noqa: E501
    return inverse_grid


def invert_nonseparable_grid(grid, input_shape):
    grid = grid.clone()
    device = grid.device
    _, _, H, W = input_shape
    B, grid_H, grid_W, _ = grid.shape
    assert B == input_shape[0]

    eps = 1e-8
    grid[:, :, :, 0] = (grid[:, :, :, 0] + 1) / 2 * (W - 1)
    grid[:, :, :, 1] = (grid[:, :, :, 1] + 1) / 2 * (H - 1)
    # grid now ranges from 0 to ([H or W] - 1)
    # TODO: implement batch operations
    inverse_grid = 2 * max(H, W) * torch.ones(
        (B, H, W, 2), dtype=torch.float32, device=device)
    for b in range(B):
        # each of these is ((grid_H - 1)*(grid_W - 1)) x 2
        p00 = grid[b,  :-1,   :-1, :].contiguous().view(-1, 2)  # noqa: 203
        p10 = grid[b, 1:  ,   :-1, :].contiguous().view(-1, 2)  # noqa: 203
        p01 = grid[b,  :-1,  1:  , :].contiguous().view(-1, 2)  # noqa: 203
        p11 = grid[b, 1:  ,  1:  , :].contiguous().view(-1, 2)  # noqa: 203

        ref = torch.floor(p00).type(torch.int)
        v00 = p00 - ref
        v10 = p10 - ref
        v01 = p01 - ref
        v11 = p11 - ref

        min_x = int(floor(min(v00[:, 0].min(), v10[:, 0].min()) - eps))
        max_x = int(ceil(max(v01[:, 0].max(), v11[:, 0].max()) + eps))
        min_y = int(floor(min(v00[:, 1].min(), v01[:, 1].min()) - eps))
        max_y = int(ceil(max(v10[:, 1].max(), v11[:, 1].max()) + eps))

        pts = torch.cartesian_prod(
            torch.arange(min_x, max_x + 1, device=device),
            torch.arange(min_y, max_y + 1, device=device),
        ).T

        # each of these is  ((grid_H - 1)*(grid_W - 1)) x 2
        vb = v10 - v00
        vc = v01 - v00
        vd = v00 - v10 - v01 + v11

        vx = pts.permute(1, 0).unsqueeze(0)  # 1 x (x_range*y_range) x 2
        Ma = v00.unsqueeze(1) - vx  # noqa: E501, ((grid_H - 1)*(grid_W - 1)) x (x_range*y_range) x 2

        vc_cross_vd = (vc[:, 0] * vd[:, 1] - vc[:, 1] * vd[:, 0]).unsqueeze(1)  # noqa: E501, ((grid_H - 1)*(grid_W - 1)) x 1
        vc_cross_vb = (vc[:, 0] * vb[:, 1] - vc[:, 1] * vb[:, 0]).unsqueeze(1)  # noqa: E501, ((grid_H - 1)*(grid_W - 1)) x 1
        Ma_cross_vd = (Ma[:, :, 0] * vd[:, 1].unsqueeze(1) - Ma[:, :, 1] * vd[:, 0].unsqueeze(1))  # noqa: E501, ((grid_H - 1)*(grid_W - 1)) x (x_range*y_range)
        Ma_cross_vb = (Ma[:, :, 0] * vb[:, 1].unsqueeze(1) - Ma[:, :, 1] * vb[:, 0].unsqueeze(1))  # noqa: E501, ((grid_H - 1)*(grid_W - 1)) x (x_range*y_range)

        qf_a = vc_cross_vd.expand(*Ma_cross_vd.shape)
        qf_b = vc_cross_vb + Ma_cross_vd
        qf_c = Ma_cross_vb

        mu_neg = -1 * torch.ones_like(Ma_cross_vd)
        mu_pos = -1 * torch.ones_like(Ma_cross_vd)
        mu_linear = -1 * torch.ones_like(Ma_cross_vd)

        nzie = (qf_a.abs() > 1e-10).expand(*Ma_cross_vd.shape)

        disc = (qf_b[nzie]**2 - 4 * qf_a[nzie] * qf_c[nzie]) ** 0.5
        mu_pos[nzie] = (-qf_b[nzie] + disc) / (2 * qf_a[nzie])
        mu_neg[nzie] = (-qf_b[nzie] - disc) / (2 * qf_a[nzie])
        mu_linear[~nzie] = qf_c[~nzie] / qf_b[~nzie]

        mu_pos_valid = torch.logical_and(mu_pos >= 0, mu_pos <= 1)
        mu_neg_valid = torch.logical_and(mu_neg >= 0, mu_neg <= 1)
        mu_linear_valid = torch.logical_and(mu_linear >= 0, mu_linear <= 1)

        mu = -1 * torch.ones_like(Ma_cross_vd)
        mu[mu_pos_valid] = mu_pos[mu_pos_valid]
        mu[mu_neg_valid] = mu_neg[mu_neg_valid]
        mu[mu_linear_valid] = mu_linear[mu_linear_valid]

        lmbda = -1 * (Ma[:, :, 1] + mu * vc[:, 1:2]) / (vb[:, 1:2] + vd[:, 1:2] * mu)  # noqa: E501

        unwarped_pts = torch.stack((lmbda, mu), dim=0)

        good_indices = torch.logical_and(
            torch.logical_and(-eps <= unwarped_pts[0],
                              unwarped_pts[0] <= 1+eps),
            torch.logical_and(-eps <= unwarped_pts[1],
                              unwarped_pts[1] <= 1+eps),
        )  # ((grid_H - 1)*(grid_W - 1)) x (x_range*y_range)
        nonzero_good_indices = good_indices.nonzero()
        inverse_j = pts[0, nonzero_good_indices[:, 1]] + ref[nonzero_good_indices[:, 0], 0]  # noqa: E501
        inverse_i = pts[1, nonzero_good_indices[:, 1]] + ref[nonzero_good_indices[:, 0], 1]  # noqa: E501
        # TODO: is replacing this with reshape operations on good_indices faster? # noqa: E501
        j = nonzero_good_indices[:, 0] % (grid_W - 1)
        i = nonzero_good_indices[:, 0] // (grid_W - 1)
        grid_mappings = torch.stack(
            (j + unwarped_pts[1, good_indices], i + unwarped_pts[0, good_indices]),  # noqa: E501
            dim=1
        )
        in_bounds = torch.logical_and(
            torch.logical_and(0 <= inverse_i, inverse_i < H),
            torch.logical_and(0 <= inverse_j, inverse_j < W),
        )
        inverse_grid[b, inverse_i[in_bounds], inverse_j[in_bounds], :] = grid_mappings[in_bounds, :]  # noqa: E501

    inverse_grid[..., 0] = (inverse_grid[..., 0]) / (grid_W - 1) * 2.0 - 1.0  # noqa: E501
    inverse_grid[..., 1] = (inverse_grid[..., 1]) / (grid_H - 1) * 2.0 - 1.0  # noqa: E501
    return inverse_grid
