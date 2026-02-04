import sys
import copy
import torch

def load_stylegan2_generators(
    stylegan2_repo_dir="third_party/stylegan2-pytorch",
    ckpt_path="weights/stylegan2-ffhq-config-f.pt",
    device=None,
    latent_dim=512,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if stylegan2_repo_dir not in sys.path:
        sys.path.insert(0, stylegan2_repo_dir)

    from model import Generator

    generator_s = Generator(size=1024, style_dim=latent_dim, n_mlp=8).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    generator_s.load_state_dict(state_dict["g_ema"], strict=True)
    generator_s.eval()
    for p in generator_s.parameters():
        p.requires_grad_(False)

    generator_t = copy.deepcopy(generator_s)
    generator_t.train()
    for p in generator_t.parameters():
        p.requires_grad_(True)

    return generator_s, generator_t, device