import argparse
import time
from utils.io import load_yaml
from types import SimpleNamespace
from utils.names_match_torch import methods
import os
from utils.common import create_code_snapshot
import numpy as np
import torch
from torchvision import models
from torchvision.transforms.functional import normalize, resize, to_tensor, to_pil_image
from torchcam.cams import CAM, GradCAM, GradCAMpp, SmoothGradCAMpp, ScoreCAM, SSCAM
from torchcam.utils import overlay_mask
from PIL import Image
import matplotlib.pyplot as plt

VGG_CONFIG = {_vgg: dict(input_layer='features', conv_layer='features')
              for _vgg in models.vgg.__dict__.keys()}

RESNET_CONFIG = {_resnet: dict(input_layer='conv1', conv_layer='layer4', fc_layer='fc')
                 for _resnet in models.resnet.__dict__.keys()}

DENSENET_CONFIG = {_densenet: dict(input_layer='features', conv_layer='features', fc_layer='classifier')
                   for _densenet in models.densenet.__dict__.keys()}

MODEL_CONFIG = {
    **VGG_CONFIG, **RESNET_CONFIG, **DENSENET_CONFIG,
    'mobilenet_v2': dict(input_layer='features', conv_layer='features')
}

def main(args):
    params = load_yaml(args.parameters)
    criterion = torch.nn.CrossEntropyLoss()
    params['verbose'] = args.verbose
    print(params, end="\n\n")
    final_params = SimpleNamespace(**params)
    time_start = time.time()
    print("Start Time",time_start)
    method = methods[final_params.method](final_params, criterion, final_params.use_cuda)
    print("Printing Method")
    print(method)
    valid_acc, elapsed, ram_usage, ext_mem_sz, preds = method.train_model(tune=False)

    mydevice = 'cuda:0' 
    cam_model_name = 'densenet161'
    
    cam_model = models.__dict__[cam_model_name](pretrained=True).eval().to(device=mydevice)

    conv_layer = MODEL_CONFIG[cam_model_name]['conv_layer']
    input_layer = MODEL_CONFIG[cam_model_name]['input_layer']
    fc_layer = MODEL_CONFIG[cam_model_name]['fc_layer']

    print("conv_layer",conv_layer)
    print("input_layer",input_layer)
    print("fc_layer",fc_layer)

    img_path = r"/home/ubuntu/CVPR20_CLVision_challenge-master/core50/data/core50_challenge_test/00a1c140473dd20e164df4b43edf839f.png"
    pil_img = Image.open(img_path, mode='r').convert('RGB')
    #pil_img = Image.open("C:\CVPR20_CLVision_challenge-master\core50\data\core50_challenge_test\000f37ff79857a7f38b945c73803f8e9.png", mode='r').convert('RGB')

    mydevice = 'cuda:0' # torch.device(args.device)
    # Preprocess image
    img_tensor = normalize(to_tensor(resize(pil_img, (224, 224))),
                           [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).to(device=mydevice)

    cam_extractors = [CAM(cam_model, conv_layer, fc_layer), GradCAM(cam_model, conv_layer),
                      GradCAMpp(cam_model, conv_layer), SmoothGradCAMpp(cam_model, conv_layer, input_layer),
                      ScoreCAM(cam_model, conv_layer, input_layer), SSCAM(cam_model, conv_layer, input_layer)]

    fig, axes = plt.subplots(1, len(cam_extractors), figsize=(7, 2))
    for idx, extractor in enumerate(cam_extractors):
        cam_model.zero_grad()
        scores = cam_model(img_tensor.unsqueeze(0))

        # Select the class index
        #class_idx = scores.squeeze(0).argmax().item() if args.class_idx is None else args.class_idx
        class_idx = scores.squeeze(0).argmax().item()
        # Use the hooked data to compute activation map
        activation_map = extractor(class_idx, scores).cpu()
        # Clean data
        extractor.clear_hooks()
        # Convert it to PIL image
        # The indexing below means first image in batch
        heatmap = to_pil_image(activation_map, mode='F')
        # Plot the result
        result = overlay_mask(pil_img, heatmap)

        axes[idx].imshow(result)
        axes[idx].axis('off')
        axes[idx].set_title(extractor.__class__.__name__, size=8)

    plt.tight_layout()
    plt.savefig("/home/ubuntu/CVPR20_CLVision_challenge-master/figure.png", dpi=200, transparent=True, bbox_inches='tight', pad_inches=0)

    # directory with the code snapshot to generate the results
    print("Creating Submission")
    sub_dir = 'submissions/' + final_params.sub_dir
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)
    # copy code
    create_code_snapshot(".", sub_dir + "/code_snapshot")

    with open(sub_dir + "/metadata.txt", "w") as wf:
        for obj in [
            np.average(valid_acc), elapsed, np.average(ram_usage),
            np.max(ram_usage), np.average(ext_mem_sz), np.max(ext_mem_sz)
        ]:
            wf.write(str(obj) + "\n")

    with open(sub_dir + "/test_preds.txt", "w") as wf:
        for pred in preds:
            wf.write(str(pred) + "\n")

    #time_end = time.time()
    elapsed2 = (time.time() - time_start) / 60
    print("Training Time: {}m".format(elapsed2))


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser('CVPR Continual Learning Challenge')
    parser.add_argument('--parameters', dest='parameters', default='config/final/default.yml')
    parser.add_argument('--verbose', type=bool, default=False,
                        help='print information or not')
    args = parser.parse_args()
    main(args)