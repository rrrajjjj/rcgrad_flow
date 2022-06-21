from tabnanny import check
import clip
import torch
import wav2clip

from .eval_utils import clean_pred
from .eval_utils import extract_audio_embeddings
from .eval_utils import preprocess, preporcess_flow
from .modules.gradcam import GradCAM
from .modules.resnet import BasicBlock
from .modules.resnet import resnet18
from .modules.resnet import ResNetSpec

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MODEL_URL = "https://github.com/hohsiangwu/rethinking-visual-sound-localization/releases/download/v0.1.0-alpha/rc_grad.pt"
checkpoint = torch.hub.load_state_dict_from_url(
            MODEL_URL, map_location=device, progress=True
        )


class RCGrad:
    def __init__(self, modal="vision", checkpoint = checkpoint, gradcam_model = "vision"):
        super(RCGrad).__init__()

        image_encoder = resnet18(modal="vision", pretrained=False)
        flow_encoder = resnet18(modal="flow", pretrained = False)
        audio_encoder = ResNetSpec(
            BasicBlock,
            [2, 2, 2, 2],
            pool="avgpool",
            zero_init_residual=False,
            groups=1,
            width_per_group=64,
            replace_stride_with_dilation=None,
            norm_layer=None,
        )


        image_encoder.load_state_dict(
            {
                k.replace("image_encoder.", ""): v
                for k, v in checkpoint.items()
                if k.startswith("image_encoder")
            }
        )
        audio_encoder.load_state_dict(
            {
                k.replace("audio_encoder.", ""): v
                for k, v in checkpoint.items()
                if k.startswith("audio_encoder")
            }
        )

        flow_encoder.load_state_dict(
            {
                k.replace("flow_encoder.", ""): v
                for k, v in checkpoint.items()
                if k.startswith("flow_encoder")
            }
        )
        self.gradcam_model = gradcam_model

        if self.gradcam_model == "vision":
            self.gradcam_encoder = image_encoder 
        elif self.gradcam_model == "flow":
            self.gradcam_encoder = flow_encoder

        target_layers = [self.gradcam_encoder.layer4[-1]]
        self.audio_encoder = audio_encoder
        self.cam = GradCAM(
            model=self.gradcam_encoder,
            target_layers=target_layers,
            use_cuda=False,
            reshape_transform=None,
        )

    def pred_audio(self, img, audio, flow = None):
        if self.gradcam_model == "vision":
            in_tensor = preprocess(img)
        if self.gradcam_model == "flow":
            in_tensor = preporcess_flow(flow)

        grayscale_cam = self.cam(
            input_tensor=in_tensor.unsqueeze(0).float(),
            targets=[self.audio_encoder(torch.from_numpy(audio).unsqueeze(0))],
        )
        pred_audio = grayscale_cam[0, :]
        pred_audio = clean_pred(pred_audio)
        return pred_audio

