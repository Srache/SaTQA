import torch
import torch.nn as nn
import torch.nn.functional as F


class NT_Xent_Sup(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.t = temperature
    
    def forward(self, z_i, z_j, label):
        N = 2 * z_i.shape[0]

        z = torch.cat([z_i, z_j], dim=0)
        # label = torch.cat([label, label], dim=0)

        # calc similarity
        z = F.normalize(z, p=2, dim=-1)
        sim = torch.mm(z, z.T) / self.t
        label = label.cpu()

        mask_pos = torch.mm(label.to_sparse(), label.T)
        mask_pos = mask_pos.fill_diagonal_(0).to(sim.device)
        mask_neg = torch.ones((N, N)).fill_diagonal_(0).to(sim.device)

        # calc nce
        pos_cnt = torch.sum(mask_pos, dim=-1)
        denominator = torch.sum(torch.exp(sim) * mask_neg, dim=-1)

        # remove the sim column which not contain same class
        idx = torch.where(pos_cnt > 0)
        numerator = torch.sum(sim * mask_pos, dim=-1)
        numerator = numerator[idx] / pos_cnt[idx]
        denominator = denominator[idx]

        loss = torch.mean(torch.log(denominator) - numerator)

        return loss




#        # remove the sim column which not contain same class
#        idx = torch.where(pos_cnt > 0)
#        numerator = torch.sum(torch.exp(sim) * mask_pos, dim=-1)
#        numerator = numerator[idx] / pos_cnt[idx]
#        denominator = denominator[idx]
#
#        loss = torch.mean(torch.log(denominator) - torch.log(numerator))
#
#        return loss


if __name__ == '__main__':
    SEED = 0
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    inp1 = torch.randn(3, 12)
    inp2 = torch.randn(3, 12)

    label = torch.tensor([[1, 0, 0, 0],
                          [1, 0, 0, 0],
                          [1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])
    
#    label = torch.tensor([[0, 1, 0, 0],
#                          [0, 1, 0, 0],
#                          [0, 0, 0, 1],
#                          [1, 0, 0, 0],
#                          [1, 0, 0, 0],
#                          [1, 0, 0, 0]])


#    inp1 = torch.randn(6, 128)
#    inp2 = torch.randn(6, 128)
#    label = torch.randn(12, 126)
    from nt_xent_sup import NT_Xent_Sup as loss_origin

    loss_fn = loss_origin(temperature=0.1)
    print(loss_fn(inp1, inp2, label))

    loss_fn = NT_Xent_Sup(temperature=0.1)
    print(loss_fn(inp1, inp2, label))

    