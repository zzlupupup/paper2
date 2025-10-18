import torch
from monai.networks.blocks import PatchEmbeddingBlock
from torch.nn import MultiheadAttention

pred = torch.rand(size=[2, 3, 256, 256]).cuda()

embed_block = PatchEmbeddingBlock(
    in_channels=3,
    img_size=[256, 256],
    patch_size=[4, 4],
    hidden_size=128,
    num_heads=8,
    spatial_dims=2
).cuda()

cross_atten = MultiheadAttention(
    embed_dim=128,
    num_heads=8,
    batch_first=True
).cuda()
Q_embeds = embed_block(pred)

#      tokens_out, _ = cross_atten(Q_DWT, K_pred, V_pred, need_weights=False)
#      输入：pred = torch.rand(size=[b, c, 256, 256])
#      tokens_out_crossatten = [2, 4096, 128]
#      decoder
#      decoder_img = [b, c, 256, 256] 
#      img = torch.cat[decoder_img, pred]   [b, 2c, 256, 256]
#      输出： img = SE(img) [b, c, 256, 256]
