import os
import torch

from .freezing import (
    freeze_all_layers,
    find_top_k_layers,
    unfreeze_weight_only,
    unfreeze_full_block,
)
from .utils import (
    show_img,
    save_img,
    save_checkpoint, tensor_to_pil
)


class Trainer:

    def __init__(self,
                 generator_s,
                 generator_t,
                 clip_loss,
                 global_loss,
                 optimizer,
                 batch_size=32,
                 z_dim=512,
                 save_img_path=None,
                 save_checkpoint_path=None,):
        self.generator_s = generator_s
        self.generator_t = generator_t
        self.clip_loss = clip_loss
        self.global_loss = global_loss
        self.optimizer = optimizer
        self.device = next(generator_t.parameters()).device
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.save_img_path = save_img_path
        self.save_checkpoint_path = save_checkpoint_path
        self.z_fixed = torch.randn(1, self.z_dim, device=self.device)

    def _generate_z(self):
        return torch.randn(self.batch_size, self.z_dim, device=self.device)

    def _apply_layer_selection(self, iter_idx,
                         top_k=12,
                         reselection_every=1,
                         unfreeze_layers_policy="weight_only", # "weight_only" | "full"
                         ):

        if iter_idx % reselection_every == 0:
            freeze_all_layers(self.generator_t)
            top_vals, top_k_layers = find_top_k_layers(self.generator_t, top_k, self.global_loss)
            if unfreeze_layers_policy == "weight_only":
                unfreeze_weight_only(self.generator_t, top_k_layers)
            elif unfreeze_layers_policy == "full":
                unfreeze_full_block(self.generator_t, top_k_layers)
            else:
                raise ValueError("unfreeze_layers_policy must be 'weight_only' or 'full'")


    def train(self,
              top_k = 12,
              steps = 500,
              reselection_every = 1,
              unfreeze_layers_policy="weight_only", # "weight_only" | "full"
              use_layer_selection=True,
              flag_save_checkpoint=True):

        extra = {
            'top_k' : top_k,
            'reselection_every' : reselection_every,
            'unfreeze_layers_policy' : unfreeze_layers_policy,
            'use_layer_selection' : use_layer_selection,
        }

        self.generator_s.eval()
        freeze_all_layers(self.generator_s)
        freeze_all_layers(self.generator_t)
        if not use_layer_selection:
            unfreeze_weight_only(self.generator_t, range(self.generator_t.n_latent - 1))

        for i in range(steps):

            if use_layer_selection:
                self._apply_layer_selection(i,
                                      top_k=top_k,
                                      reselection_every=reselection_every,
                                      unfreeze_layers_policy=unfreeze_layers_policy)

            self.generator_t.train()
            if hasattr(self.generator_t, "style"):
                self.generator_t.style.eval()
            self.optimizer.zero_grad()

            z = self._generate_z()

            with torch.no_grad():
                source_img, _ = self.generator_s([z], randomize_noise=False)
            target_img, _ = self.generator_t([z], randomize_noise=False)

            loss = self.clip_loss(source_img, target_img)
            loss.backward()
            self.optimizer.step()

            if i % 100 == 0:
                print(i, loss.item())

                with torch.no_grad():
                    target_img, _ = self.generator_t([self.z_fixed], input_is_latent=False, randomize_noise=False)

                    img_pil = tensor_to_pil(target_img)
                    path = os.path.join(self.save_img_path, f"step_{i:06d}.png")
                    save_img(target_img, path)
                    show_img(img_pil)
            if flag_save_checkpoint:
                if i % 200 == 0:
                    ckpt_path = os.path.join(self.save_checkpoint_path, f"ckpt_{i:06d}.pt")
                    save_checkpoint(path=ckpt_path,
                                    step=i,
                                    generator_t=self.generator_t,
                                    optimizer=self.optimizer,
                                    extra=extra)

        ckpt_path = os.path.join(self.save_checkpoint_path, f"ckpt_{steps:06d}.pt")
        save_checkpoint(path=ckpt_path,
                        step=steps,
                        generator_t=self.generator_t,
                        optimizer=self.optimizer)