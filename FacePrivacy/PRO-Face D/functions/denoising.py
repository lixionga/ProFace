import torch
from tqdm import tqdm
import torchvision.utils as tvu
import torchvision
import os

from .arcface_torch.model import IDLoss


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def arcface_ddim_diffusion_privacy_protection(
        x, seq, model, b,
        cls_fn=None, rho_scale=1.0, stop=100, ref_path=None,
        tao=0, energy_fun='res', model_kwargs=None):
    # print("      5.2.0 load IDLoss...")
    idloss = IDLoss(ref_path=ref_path).cuda()

    # setup iteration variables
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]
    # print("      5.2.1 sample begin...")
    # iterate over the timesteps
    for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        at = compute_alpha(b, t.long())  # ā_{t}
        at_next = compute_alpha(b, next_t.long())  # ā_{t-1}
        xt = xs[-1].to('cuda:0')
        # print(f"         i = {i}, j = {j}...")
        xt.requires_grad = True

        if cls_fn == None:
            if i <= tao:
                cond = model_kwargs['cond']
                model_kwargs = {'cond': torch.zeros_like(cond, requires_grad=False)}
            model_forward = model.forward(x=xt, t=t, **model_kwargs)  # 改动
            et = model_forward.pred
        else:
            print("use class_num")
            class_num = 281
            classes = torch.ones(xt.size(0), dtype=torch.long, device=torch.device("cuda:0")) * class_num
            et = model(xt, t, classes)
            et = et[:, :3]
            et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)

        if et.size(1) == 6:
            et = et[:, :3]

        # xt -> x0|t
        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

        # Time-Dependent Energy Function
        residual = None
        if energy_fun == 'res':
            residual = idloss.get_residual(x0_t)
        elif energy_fun == 'sim':
            residual = idloss.get_sim_residual(x0_t)
        elif energy_fun == 'idis':
            residual = idloss.get_idis_residual(x0_t)
        else:
            print('!!energy_fun error...')

        norm = torch.linalg.norm(residual)
        print(f"......x0_t.requires_grad={x0_t.requires_grad},norm.requires_grad={norm.requires_grad},xt.requires_grad={xt.requires_grad}")
        norm_grad = torch.autograd.grad(outputs=norm, inputs=xt)[0]  # gt
        # print(f"         norm_grad = {norm_grad}")
        eta = 0.5
        c1 = (1 - at_next).sqrt() * eta
        c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
        xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x0_t) + c2 * et

        # use guided gradient
        # ρt
        rho = at.sqrt() * rho_scale
        # x_{t-1} = x_{t-1} - gt * ρt
        if not i <= stop:
            # print(f"norm_grad={norm_grad}, i={i}")
            xt_next -= rho * norm_grad

        x0_t = x0_t.detach()
        xt_next = xt_next.detach()

        x0_preds.append(x0_t.to('cpu'))
        xs.append(xt_next.to('cpu'))
    # print("      5.2.2 sample end...")
    return [xs[-1]], [x0_preds[-1]]