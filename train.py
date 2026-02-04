
import os
import sys
import argparse
import copy
import yaml
import clip
import torch
import torch.optim as optim


def deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)  # копия, чтобы не мутировать исходный base
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_prompts(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if isinstance(data, list):
        return data
    return data["prompts"]


def to_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(x)

def main():
    p = argparse.ArgumentParser()

    p.add_argument("--base", required=True)

    p.add_argument("--domain", required=True)

    p.add_argument("--hyper", required=True)

    p.add_argument("--outputs_root", default=None)

    args = p.parse_args()

    cfg = load_yaml(args.base)
    cfg = deep_merge(cfg, load_yaml(args.domain))
    cfg = deep_merge(cfg, load_yaml(args.hyper))

    if args.outputs_root is not None:
        cfg.setdefault("paths", {})
        cfg["paths"]["outputs_root"] = args.outputs_root

    from project.utils import seed_everything
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)

    from project.configs.scripts.load_generators import load_stylegan2_generators

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = os.path.join(cfg["paths"]["weights_dir"], "stylegan2-ffhq-config-f.pt")
    generator_s, generator_t, device = load_stylegan2_generators(
        stylegan2_repo_dir=cfg["paths"]["stylegan2_repo"],
        ckpt_path=ckpt_path,
        device=device,
        latent_dim=int(cfg["train"].get("z_dim", 512)),
    )

    src_prompts = load_prompts(cfg["prompts"]["src"])
    tgt_prompts = load_prompts(cfg["prompts"]["tgt"])
    global_prompts = load_prompts(cfg["prompts"]["global"])

    tr = cfg["train"]

    from project.losses import DirectionalCLIPLoss, GlobalCLIPLoss
    from project.trainers import Trainer

    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)

    clip_loss = DirectionalCLIPLoss(src_prompts, tgt_prompts, clip_model=clip_model, device=device)

    global_loss = GlobalCLIPLoss(global_prompts, clip_model=clip_model, device=device)

    lr = float(tr.get("lr", 2e-3))
    optimizer = optim.Adam(generator_t.parameters(), lr=lr)

    domain_name = cfg.get("name", "exp")

    run_name = (
        f"steps{tr['steps']}_bs{tr['batch_size']}_topk{tr['top_k']}_"
        f"sel{int(to_bool(tr['use_layer_selection']))}_pol{tr['unfreeze_layers_policy']}_seed{seed}"
    )

    out_root = cfg["paths"]["outputs_root"]
    out_dir = os.path.join(out_root, domain_name, run_name)

    img_dir = os.path.join(out_dir, "images")
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    log = cfg.get("logging", {})
    trainer = Trainer(
        generator_s=generator_s,
        generator_t=generator_t,
        clip_loss=clip_loss,
        global_loss=global_loss,
        optimizer=optimizer,
        batch_size=int(tr["batch_size"]),
        z_dim=int(tr.get("z_dim", 512)),
        save_img_path=img_dir,
        save_checkpoint_path=ckpt_dir,
        preview_every=int(log.get("preview_every", 100)),
        ckpt_every=int(log.get("ckpt_every", 200))
    )

    trainer.train(
        top_k=int(tr["top_k"]),
        steps=int(tr["steps"]),
        reselection_every=int(tr["reselection_every"]),
        unfreeze_layers_policy=str(tr["unfreeze_layers_policy"]),
        use_layer_selection=to_bool(tr["use_layer_selection"]),
        flag_save_checkpoint=True,
    )


if __name__ == "__main__":
    main()