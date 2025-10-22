import torch
import torch.nn.functional as F

def get_uncertain_atten_mask(pred_l, pred_r, threshold):
    
    pred_l = torch.softmax(pred_l, dim=1)
    pred_r = torch.softmax(pred_r, dim=1)
    
    #B 1 64 64 64
    uncertain_l = -torch.sum(pred_l * torch.log(pred_l + 1e-8), dim=1)
    uncertain_r = -torch.sum(pred_r * torch.log(pred_r + 1e-8), dim=1)

    #B 4096
    B = pred_l.shape[0]
    uncertain_l_seq = F.avg_pool3d(uncertain_l, kernel_size=4, stride=4).contiguous().view(B, -1) > threshold
    uncertain_r_seq = F.avg_pool3d(uncertain_r, kernel_size=4, stride=4).contiguous().view(B, -1) > threshold

    return uncertain_l_seq, uncertain_r_seq


