import numpy as np
import pytorch_lightning as pl
import torch
from torch import cosine_similarity, nn

from ..modules.resnet import BasicBlock
from ..modules.resnet import resnet18
from ..modules.resnet import ResNetSpec


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CLIPLoss1D(nn.Module):
    def __init__(self):
        super(CLIPLoss1D, self).__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_image = nn.CrossEntropyLoss()
        self.loss_audio = nn.CrossEntropyLoss()
        self.loss_flow = nn.CrossEntropyLoss()

    def forward(self, image_features, audio_features, flow_features):
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)
        flow_features = flow_features / flow_features.norm(dim=-1, keepdim=True)

        batch_size = image_features.shape[0]
        ground_truth = torch.arange(batch_size, dtype=torch.long, device=device)
        
        # calculate pairwise losses
        # image-audio
        logits1, logits2 = self.cosine_similarity(image_features, audio_features)
        loss_img_aud = (self.loss_image(logits1, ground_truth)
                        + self.loss_audio(logits2, ground_truth)
                        ) / 2


        # image-flow
        logits1, logits2 = self.cosine_similarity(image_features, flow_features)
        loss_img_flow = (self.loss_image(logits1, ground_truth)
                        + self.loss_flow(logits2, ground_truth)
                        ) / 2
        
        # flow-audio
        logits1, logits2 = self.cosine_similarity(flow_features, audio_features)
        loss_flow_aud = (self.loss_flow(logits1, ground_truth)
                        + self.loss_audio(logits2, ground_truth)
                        ) / 2

        return ((loss_img_aud + 
                loss_img_flow + 
                loss_flow_aud) / 3, 
                [loss_img_aud, loss_img_flow, loss_flow_aud])
    


    def cosine_similarity(self, features1, features2):
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits1 = logit_scale * features1 @ features2.t()
        logits2 = logit_scale * features2 @ features1.t()

        return logits1, logits2

        
        


class LightningBase(pl.LightningModule):
    def training_step(self, batch, batch_idx):
        loss, pairwise_losses = self.step(batch, batch_idx)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_loss_img-aud", pairwise_losses[0], on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_loss_img-flow", pairwise_losses[1], on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_loss_flow-aud", pairwise_losses[2], on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        
        return {"loss": loss, "log": {"train_loss": loss}}

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("val_loss", avg_loss, prog_bar=True)   
        return {
            "avg_val_loss": avg_loss,
            "log": {"val_loss": avg_loss},
            "progress_bar": {"val_loss": avg_loss},
        }


    def test_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        return {"test_loss": loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        self.log("test_loss", avg_loss, prog_bar=True)
        return {
            "avg_test_loss": avg_loss,
            "log": {"test_loss": avg_loss},
            "progress_bar": {"test_loss": avg_loss},
        }

    def configure_optimizers(self):
        if self.args["optimizer"] == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=self.args.learning_rate, momentum=0.9
            )
        elif self.args["optimizer"] == "Adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.args["learning_rate"]
            )
        else:
            assert False
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=self.args["lr_scheduler_patience"],
            min_lr=1e-6,
            verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }


class RCGrad(LightningBase):
    def __init__(
        self,
        args,
    ):
        super().__init__()
        self.args = args
        self.model_url = args["model_url"]
        self.image_encoder = resnet18(modal="vision", pretrained=True)
        self.flow_encoder = resnet18(modal="flow", pretrained = True)
        self.audio_encoder = ResNetSpec(
            BasicBlock,
            [2, 2, 2, 2],
            pool="avgpool",
            zero_init_residual=False,
            groups=1,
            width_per_group=64,
            replace_stride_with_dilation=None,
            norm_layer=None,
        )
        self.loss_fn = CLIPLoss1D()

        if self.model_url:
            checkpoint = torch.hub.load_state_dict_from_url(
                        self.model_url, map_location=device, progress=True
                        )
            

            self.image_encoder.load_state_dict(
                {
                    k.replace("image_encoder.", ""): v
                    for k, v in checkpoint.items()
                    if k.startswith("image_encoder")
                }
            )
            self.audio_encoder.load_state_dict(
                {
                    k.replace("audio_encoder.", ""): v
                    for k, v in checkpoint.items()
                    if k.startswith("audio_encoder")
                }
            )
        

    def forward(self, audio, image, flow):
        audio_output = self.audio_encoder(audio.float())
        image_output = self.image_encoder(image.float())
        flow_output = self.flow_encoder(flow.float())
        return audio_output, image_output, flow_output

    def step(self, batch, batch_idx):
        audio, images, flow = batch
        audio_out, image_out, flow_out = self.forward(audio, images, flow)
        loss, pairwise_losses = self.loss_fn(audio_out, image_out, flow_out)
        return loss, pairwise_losses
