import torch
from torch import nn
from monai.networks.blocks import PatchEmbeddingBlock

class Fusion_Preds(nn.Module):
    
    def __init__(self, img_size=[64, 64, 64], patch_size=[4, 4, 4], fea_dim=128, num_heads=8, in_chan=3):
        super().__init__()

        self.patch_size = patch_size
        self.patch_embed = PatchEmbeddingBlock(
            in_channels=in_chan,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=fea_dim,
            num_heads=num_heads
        )

        self.Q_norm = nn.LayerNorm(fea_dim)
        self.KV_norm = nn.LayerNorm(fea_dim)
        self.ff_norm = nn.LayerNorm(fea_dim)

        self.ffn = nn.Sequential(
            nn.Linear(fea_dim, fea_dim // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fea_dim // 4, fea_dim),
            nn.Dropout(0.1)
        )

        self.MHA = nn.MultiheadAttention(
            embed_dim=fea_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )

        self.up = nn.Upsample(scale_factor=patch_size[0], mode='trilinear')

        decoder_in_channels = fea_dim + in_chan

        self.decoder = nn.Sequential(
            nn.Conv3d(decoder_in_channels, fea_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(fea_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(fea_dim // 2, fea_dim // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(fea_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(fea_dim // 4, in_chan, kernel_size=1)
        )


    def forward(self, pred_l, pred_r):

        embed_l = self.patch_embed(pred_l)
        embed_r = self.patch_embed(pred_r)

        Q = self.Q_norm(embed_l)
        K = self.KV_norm(embed_r)
        V = K

        MHA_tokens, _ = self.MHA(Q, K, V, need_weights=False)
        MHA_tokens = self.ff_norm(MHA_tokens)
        tokens = self.ffn(MHA_tokens)

        B, N, C = tokens.shape
        patch_dim = round(N ** (1/3)) 
        tokens_reshaped = tokens.permute(0, 2, 1).contiguous().view(B, C, patch_dim, patch_dim, patch_dim)
        tokens_up = self.up(tokens_reshaped)

        pred_cat = torch.cat([tokens_up, pred_r], dim=1)
        pred = self.decoder(pred_cat)

        return pred
    

