import torch
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, cfg, temp):
        super().__init__()
        self.temp = temp
        self.cfg = cfg

    def similarity(self, a, b):
        a_norm = a / a.norm(dim=1)[:, None]
        b_norm = b / b.norm(dim=1)[:, None]
        cos_sim = torch.mm(a_norm, b_norm.transpose(0, 1))
        return torch.exp(cos_sim / self.temp)

    def get_hard_cos(self, x, cos_sim, neg_y, device, num=10):
        min_num = torch.min(torch.sum(neg_y, dim=1))
        if min_num < 5:
            return torch.tensor(0).to(device)
        cos_idx = (cos_sim * neg_y).topk(5)[1]
        total_cos = torch.zeros(x.size(0), 1).to(device)
        total_hard = []
        for _ in range(num):
            rand_mat = torch.rand(x.size(0), 5)
            k_th_quant = torch.topk(rand_mat, 2, largest=False)[0][:, -1:]
            mask = rand_mat <= k_th_quant
            while torch.sum(torch.sum(mask, 1) != 2):
                rand_mat = torch.rand(x.size(0), 5)
                k_th_quant = torch.topk(rand_mat, 2, largest=False)[0][:, -1:]
                mask = rand_mat <= k_th_quant
            feature_left = x[cos_idx[mask].view(-1, 2)[:, 0]]
            feature_right = x[cos_idx[mask].view(-1, 2)[:, 1]]
            cos_left = cos_sim.gather(1, cos_idx[mask].view(-1, 2)[:, 0].view(-1,1))
            cos_right = cos_sim.gather(1, cos_idx[mask].view(-1, 2)[:, 1].view(-1,1))
            alpha = cos_left / (cos_left + cos_right)
            mixed_hard = alpha * feature_left + (1 - alpha) * feature_right
            mixed_hard = torch.nn.functional.normalize(mixed_hard)
            x = torch.nn.functional.normalize(x)
            cos = torch.exp(
                torch.nn.CosineSimilarity()(x, mixed_hard) / 0.05).unsqueeze(-1)
            total_cos += cos
            total_hard.append(mixed_hard[0].view(1,-1))

        return total_cos, total_hard

    def forward(self, x, y, hard_mode, hard_num, device):
        cos_sim = self.similarity(x, x)
        positive_pair_nums = torch.sum(y, dim=-1)
        neg_y = 1.0 - y
        num = torch.min(torch.sum(neg_y, dim=1))

        if hard_mode == 'normal':
            # nl = torch.sum((cos_sim * neg_y).topk(int(num.data * 0.2))[0], dim=-1).unsqueeze(-1)
            nl = torch.sum((cos_sim * neg_y), dim=-1).unsqueeze(-1)
        elif hard_mode == 'union':
            syn_hard, _ = self.get_hard_cos(x, cos_sim, neg_y, device, hard_num)
            nl = torch.sum((cos_sim * neg_y), dim=-1).unsqueeze(-1)
        elif hard_mode == 'single_hard':
            syn_hard, _ = self.get_hard_cos(x, cos_sim, neg_y, device, hard_num)
            nl = syn_hard
        elif hard_mode == 'no':
            return torch.tensor(0).to(device)
        fm = 1.0 / (cos_sim + nl)
        sm = torch.log(cos_sim * fm)
        lm = torch.sum(sm * y, dim=-1) / positive_pair_nums
        if hard_mode == 'union':
            new_fm = 1.0 / (cos_sim + syn_hard)
            new_sm = torch.log(cos_sim * new_fm)
            new_lm = torch.sum(new_sm * y, dim=-1) / positive_pair_nums
            lm = lm + new_lm
        # sm = torch.log(pl / (nl + pl))

        return -torch.mean(lm)
        # return -torch.mean(sm)
