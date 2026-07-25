import torch


def jigsaw(x, patch_size: int):

    B, C, H, W = x.shape
    h_patch = H // patch_size
    w_patch = W // patch_size

    patches = x.unfold(2, h_patch, h_patch).unfold(3, w_patch, w_patch)

    patches = patches.contiguous().view(B, C, -1, h_patch, w_patch)
    patches = patches.permute(0, 2, 1, 3, 4)  # [B, N_patches, C, h, w]

    num_patches = patch_size * patch_size
    rand_indices = torch.rand(B, num_patches, device=x.device).argsort(dim=1)

    shuffled_patches = patches[torch.arange(B).unsqueeze(1), rand_indices]

    shuffled_x = shuffled_patches.view(B, patch_size, patch_size, C, h_patch, w_patch)
    shuffled_x = shuffled_x.permute(0, 3, 1, 4, 2, 5).contiguous()
    shuffled_x = shuffled_x.view(B, C, H, W)

    return shuffled_x
