from functools import partial
import time
import torch
import butane

if __name__ == "__main__":
    dev = torch.device('cuda')

    pemb = butane.nn.SinusoidalEmbeddings(10)
    x = torch.rand(10, 100, 10)
    emb = x + pemb(x)

    attention = butane.nn.SelfAttention(10, n_heads = 5)
    print(attention)

    attention_mask = torch.zeros((1, 10, 20, 20))
    print(attention(emb))

    pemb = butane.nn.PatchEmbeddings(
        input_dims = (100,20,20), 
        d_model = 10, 
        patch_size = (10, 10)
    )
    emb = pemb(torch.rand(1, 100, 20, 20))
    print(emb.size())


    start_t = time.monotonic()
    local_attention = butane.nn.SpatialSelfAttention(
        64,
        n_heads = 2,
        kernel_size = 3,
        prenorm=partial(torch.nn.GroupNorm, num_groups=32)
    )
    inpu = torch.randn(1, 64, 20, 20)
    print(local_attention(inpu))
    end_t = time.monotonic()
    print(end_t - start_t)
