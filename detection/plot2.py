import os
import sys
import torch
import mmcv
from mmcv import Config
from mmdet.apis import inference_detector, show_result_pyplot
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from tqdm import tqdm

config_path = '/home/pxh/OverLoCK_youhua/detection/work_dirs/mask_rcnn_overlock_t_in1k_fpn_1x_coco/mask_rcnn_overlock_t_in1k_fpn_1x_coco.py'
checkpoint_path = '/home/pxh/OverLoCK_youhua/detection/work_dirs/mask_rcnn_overlock_t_in1k_fpn_1x_coco/latest.pth'
input_dir = '/home/ac/data/pxh/dataset_pxh/dataset_coco/test2017/'
output_dir = './vis_overlock'

model_py_path = '/home/pxh/OverLoCK_youhua/detection/models'

if model_py_path not in sys.path:
    sys.path.insert(0, model_py_path)
import overlock

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'Kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    cfg = Config.fromfile(config_path)
    if 'deploy' in cfg.model.backbone:
        cfg.model.backbone.deploy = False
        print("Config: Set backbone deploy=False")

    print("Building model...")
    model = build_detector(cfg.model)
    model.CLASSES = COCO_CLASSES
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    load_checkpoint(model, checkpoint_path, map_location='cpu', strict=False)
    
    model.cfg = cfg
    model.to(device)
    model.eval()

    img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    imgs = [f for f in os.listdir(input_dir) if f.lower().endswith(img_extensions)]
    print(f"Found {len(imgs)} images in {input_dir}")

    print("Starting inference...")
    for img_name in tqdm(imgs):
        img_path = os.path.join(input_dir, img_name)
        out_path = os.path.join(output_dir, img_name)

        with torch.no_grad():
            result = inference_detector(model, img_path)

        model.show_result(
            img_path,
            result,
            score_thr=0.3,
            show=False,
            out_file=out_path,
            bbox_color=(72, 101, 241),
            text_color=(72, 101, 241),
            mask_color=None
        )

    print(f"\nAll results are saved in: {os.path.abspath(output_dir)}")

if __name__ == '__main__':
    main()