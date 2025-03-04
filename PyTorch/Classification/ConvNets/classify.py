# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from PIL import Image
import argparse
import numpy as np
import json
import torch
from torch.cuda.amp import autocast
import torch.backends.cudnn as cudnn
import nvtx

from image_classification import models
import torchvision.transforms as transforms

from image_classification.models import (
    resnet50,
    resnext101_32x4d,
    se_resnext101_32x4d,
    efficientnet_b0,
    efficientnet_b4,
    efficientnet_widese_b0,
    efficientnet_widese_b4,
    efficientnet_quant_b0,
    efficientnet_quant_b4,
)

import torch.cuda.profiler as profiler
import pyprof  
pyprof.init()  

class AnnotatedResNet50(torch.nn.Module):
    name = 'AnnotatedResNet50'

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = resnet50(*args, **kwargs).cuda()

    def _forward_impl(self, x):
    # Stem
        start = nvtx.start_range(message="Stem", color="blue")
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        torch.cuda.synchronize()
        nvtx.end_range(start)
        
        # Layer 1 (first stage: conv2_x)
        start = nvtx.start_range(message="Layer1", color="green")
        x = self.model.layers[0](x)  # layers[0] is equivalent to layer1
        torch.cuda.synchronize()
        nvtx.end_range(start)

        # Layer 25 (approximation: conv3_x and conv4_x)
        start = nvtx.start_range(message="Layer25", color="red")
        x = self.model.layers[1](x)  # layers[1] is equivalent to layer2
        x = self.model.layers[2](x)  # layers[2] is equivalent to layer3
        torch.cuda.synchronize()
        nvtx.end_range(start)

        # Layer 50 (last stage: conv5_x)
        start = nvtx.start_range(message="Layer50", color="orange")
        x = self.model.layers[3](x)  # layers[3] is equivalent to layer4
        torch.cuda.synchronize()
        nvtx.end_range(start)

        # Final Classifier
        start = nvtx.start_range(message="Classifier", color="purple")
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        torch.cuda.synchronize()
        nvtx.end_range(start)
    
        return x
    
    def forward(self, x):
        return self._forward_impl(x)

    @classmethod
    def parser(cls):
        return resnet50.parser()

def available_models():
    models = {
        m.name: m
        for m in [
            resnet50,
            resnext101_32x4d,
            se_resnext101_32x4d,
            efficientnet_b0,
            efficientnet_b4,
            efficientnet_widese_b0,
            efficientnet_widese_b4,
            efficientnet_quant_b0,
            efficientnet_quant_b4,
            AnnotatedResNet50,
        ]
    }
    return models


def add_parser_arguments(parser):
    model_names = available_models().keys()
    parser.add_argument("--image-size", default="224", type=int)
    parser.add_argument(
        "--arch",
        "-a",
        metavar="ARCH",
        default="resnet50",
        choices=model_names,
        help="model architecture: " + " | ".join(model_names) + " (default: resnet50)",
    )
    parser.add_argument(
        "--precision", metavar="PREC", default="AMP", choices=["AMP", "FP32"]
    )
    parser.add_argument("--cpu", action="store_true", help="perform inference on CPU")
    parser.add_argument("--image", metavar="<path>", help="path to classified image")


def load_jpeg_from_file(path, image_size, cuda=True):
    img_transforms = transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )

    img = img_transforms(Image.open(path))
    with torch.no_grad():
        # mean and std are not multiplied by 255 as they are in training script
        # torch dataloader reads data into bytes whereas loading directly
        # through PIL creates a tensor with floats in [0,1] range
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        if cuda:
            mean = mean.cuda()
            std = std.cuda()
            img = img.cuda()
        img = img.float()

        input = img.unsqueeze(0).sub_(mean).div_(std)

    return input


def check_quant_weight_correctness(checkpoint_path, model):
    state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    state_dict = {
        k[len("module.") :] if k.startswith("module.") else k: v
        for k, v in state_dict.items()
    }
    quantizers_sd_keys = {
        f"{n[0]}._amax" for n in model.named_modules() if "quantizer" in n[0]
    }
    sd_all_keys = quantizers_sd_keys | set(model.state_dict().keys())
    assert set(state_dict.keys()) == sd_all_keys, (
        f"Passed quantized architecture, but following keys are missing in "
        f"checkpoint: {list(sd_all_keys - set(state_dict.keys()))}"
    )


def main(args, model_args):
    imgnet_classes = np.array(json.load(open("./LOC_synset_mapping.json", "r")))
    try:
        model = available_models()[args.arch](**model_args.__dict__)
    except RuntimeError as e:
        print_in_box(
            "Error when creating model, did you forget to run checkpoint2model script?"
        )
        raise e

    if args.arch in ["efficientnet-quant-b0", "efficientnet-quant-b4"]:
        check_quant_weight_correctness(model_args.pretrained_from_file, model)

    if not args.cpu:
        model = model.cuda()
    model.eval()

    input = load_jpeg_from_file(args.image, args.image_size, cuda=not args.cpu)

    with torch.no_grad(), autocast(enabled=args.precision == "AMP"):
        profiler.start()
        output = model(input)
        profiler.stop()
        output = torch.nn.functional.softmax(output, dim=1)

    output = output.float().cpu().view(-1).numpy()
    top5 = np.argsort(output)[-5:][::-1]

    print(args.image)
    for c, v in zip(imgnet_classes[top5], output[top5]):
        print(f"{c}: {100*v:.1f}%")


def print_in_box(msg):
    print("#" * (len(msg) + 10))
    print(f"#### {msg} ####")
    print("#" * (len(msg) + 10))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch ImageNet Classification")

    add_parser_arguments(parser)
    args, rest = parser.parse_known_args()
    model_args, rest = available_models()[args.arch].parser().parse_known_args(rest)

    assert len(rest) == 0, f"Unknown args passed: {rest}"

    cudnn.benchmark = True

    main(args, model_args)
