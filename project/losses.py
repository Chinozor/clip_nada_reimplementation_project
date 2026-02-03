import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import kornia.augmentation as K

class _BaseCLIPLoss(nn.Module):
    def __init__(self, device='cuda', eps=1e-6, clip_model=None, clip_name='ViT-B/32'):
        super(_BaseCLIPLoss, self).__init__()

        if clip_model is None:
            self.model, _ = clip.load(clip_name, device=device)
        else:
            self.model = clip_model

        self.aug = K.AugmentationSequential(
            K.RandomResizedCrop((224, 224), scale=(0.8, 1.0), ratio=(0.9, 1.1), p=1.0),
            K.RandomHorizontalFlip(p=0.5),
            K.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05, p=0.8),
            same_on_batch=False
        )

        self.device = next(self.model.parameters()).device
        self.aug = self.aug.to(self.device)
        self.eps = eps

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device).view(1, 3, 1, 1)
        self.register_buffer("clip_mean", mean, persistent=False)
        self.register_buffer("clip_std", std, persistent=False)

    def _prep_image(self, image):
        image = ((image + 1) / 2).clamp(0, 1).float()
        image = F.interpolate(image, size=(256, 256), mode="bilinear", align_corners=False)
        return image

    def _encode_text(self, prompts):

        if isinstance(prompts, str):
            prompts = [prompts]

        with torch.no_grad():
            t_feat = self.model.encode_text(clip.tokenize(prompts).to(self.device)).float()
            t_feat = F.normalize(t_feat, dim=-1, eps=self.eps)
            t_feat = t_feat.mean(dim=0)
            t_feat = F.normalize(t_feat, dim=-1, eps=self.eps)

        return t_feat



class GlobalCLIPLoss(_BaseCLIPLoss):
    def __init__(self, tgt_prompts, device='cuda', eps=1e-6, clip_model=None, clip_name='ViT-B/32'):
        super(GlobalCLIPLoss, self).__init__(device=device, eps=eps, clip_model=clip_model, clip_name=clip_name)

        t_feat = self._encode_text(tgt_prompts)
        self.register_buffer("t_feat",  t_feat,  persistent=True)

    def forward(self, image_t):
        image_t = self._prep_image(image_t)
        image_t = self.aug(image_t)
        image_t = (image_t - self.clip_mean) / self.clip_std

        image_t_features = self.model.encode_image(image_t).float()

        image_t_features = F.normalize(image_t_features, dim=-1, eps=self.eps)
        similarity = F.cosine_similarity(image_t_features, self.t_feat, dim=-1)

        loss = 1 - similarity

        return loss.mean()


class DirectionalCLIPLoss(_BaseCLIPLoss):
    def __init__(self, src_prompts, tgt_prompts, device='cuda', eps=1e-6, clip_model=None, clip_name='ViT-B/32'):
        super(DirectionalCLIPLoss, self).__init__(device=device, eps=eps, clip_model=clip_model, clip_name=clip_name)

        e_src = self._encode_text(src_prompts)
        e_tgt = self._encode_text(tgt_prompts)

        with torch.no_grad():
            t_dist = e_tgt - e_src
            t_dist = F.normalize(t_dist, dim=-1, eps=self.eps)
        self.register_buffer("t_dist",  t_dist,  persistent=True)

    def forward(self, image_s, image_t):

        with torch.no_grad():
            image_s = self._prep_image(image_s)
            image_s = self.aug(image_s)
            params = self.aug._params
            image_s = (image_s - self.clip_mean) / self.clip_std
            image_s_features = self.model.encode_image(image_s).float()
            image_s_features = F.normalize(image_s_features, dim=-1, eps=self.eps)

        image_t = self._prep_image(image_t)  # [B, 3, 224, 224]
        image_t = self.aug(image_t, params=params)
        image_t = (image_t - self.clip_mean) / self.clip_std

        image_t_features = self.model.encode_image(image_t).float()

        image_t_features = F.normalize(image_t_features, dim=-1, eps=self.eps)

        delta_image_features = image_t_features - image_s_features

        delta_image_features = F.normalize(delta_image_features, dim=-1, eps=self.eps)

        similarity = (delta_image_features * self.t_dist).sum(dim=-1)

        loss = 1 - similarity

        return loss.mean()