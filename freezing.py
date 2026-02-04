import torch
import torch.nn as nn
import torch.optim as optim

def find_top_k_layers(model, top_k, g_loss, num_e = 1):

    top_k = min(top_k, model.n_latent - 1)

    device = next(model.parameters()).device
    z = torch.randn(8, 512, device=device)

    with torch.no_grad():
        w = model.style(z) # [8, 512]

    w_plus_0 = w.unsqueeze(1).repeat(1, model.n_latent, 1)
    w_plus = torch.nn.Parameter(w_plus_0.clone())
    opt = optim.Adam([w_plus], lr=0.25)

    for _ in range(num_e):
        opt.zero_grad()
        img, _ = model([w_plus], input_is_latent=True, randomize_noise=False)
        loss = g_loss(img)
        loss.backward()
        opt.step()

    diff = (w_plus - w_plus_0).detach()

    delta = diff.norm(dim=-1).mean(dim=0)

    top_vals, top_k_layers = torch.topk(delta[:-1], k=top_k, largest=True)

    return top_vals, top_k_layers

def _as_int(k):
    return int(k.item()) if hasattr(k, "item") else int(k)

@torch.no_grad()
def unfreeze_weight_only(model, top_k_layers):
    for k in top_k_layers:
        k = _as_int(k)
        if k == 0:
            model.conv1.conv.weight.requires_grad_(True)
        elif k != model.n_latent - 1:
            model.convs[k-1].conv.weight.requires_grad_(True)

@torch.no_grad()
def unfreeze_full_block(model, top_k_layers):
    for k in top_k_layers:
        k = _as_int(k)
        if k == 0:
            model.conv1.requires_grad_(True)
        elif k != model.n_latent - 1:
            model.convs[k-1].requires_grad_(True)

@torch.no_grad()
def freeze_all_layers(model):
    for p in model.parameters():
        p.requires_grad_(False)