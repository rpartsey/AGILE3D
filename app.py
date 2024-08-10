import argparse
import json
from collections import defaultdict

import numpy as np
from utils.ply import read_ply

import torch
from interactive_tool.utils import *
from interactive_tool.interactive_segmentation_user import UserInteractiveSegmentationModel
from interactive_tool.dataloader import InteractiveDataLoader

from flask import Flask
from flask import request
from flask_cors import CORS


app = Flask(__name__)
CORS(app)


def get_default_config():
    parser = argparse.ArgumentParser()

    # minimal arguments:
    parser.add_argument('--user_name', type=str, default='user_00')
    parser.add_argument('--pretraining_weights', type=str,
                        default='weights/checkpoint1099.pth')
    parser.add_argument('--dataset_scenes', type=str,
                        default='data/interactive_dataset')
    parser.add_argument('--point_type', type=str, default=None, help="choose between 'mesh' and 'pointcloud'. If not given, the type will be determined automatically")
    
    # model
    ### 1. backbone
    parser.add_argument('--dialations', default=[ 1, 1, 1, 1 ], type=list)
    parser.add_argument('--conv1_kernel_size', default=5, type=int)
    parser.add_argument('--bn_momentum', default=0.02, type=int)
    parser.add_argument('--voxel_size', default=0.05, type=float)

    ### 2. transformer
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--dim_feedforward', default=1024, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--num_decoders', default=3, type=int)
    parser.add_argument('--num_bg_queries', default=10, type=int, help='number of learnable background queries')
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--pre_norm', default=False, type=bool)
    parser.add_argument('--normalize_pos_enc', default=True, type=bool)
    parser.add_argument('--positional_encoding_type', default="fourier", type=str)
    parser.add_argument('--gauss_scale', default=1.0, type=float, help='gauss scale for positional encoding')
    parser.add_argument('--hlevels', default=[4], type=list)
    parser.add_argument('--shared_decoder', default=False, type=bool)
    parser.add_argument('--aux', default=True, type=bool, help='whether supervise layer by layer')
    
    parser.add_argument('--device', default='cuda')
    
    config = parser.parse_args()

    return config


config = get_default_config()
config.user_name = "test_user"
config.pretraining_weights = "weights/model.pth" 
config.dataset_scenes = "data/interactive_dataset"

device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
dataloader_test = InteractiveDataLoader(config)
model = UserInteractiveSegmentationModel(device, config, dataloader_test)
print(f"Using {device}")
model.run_segmentation()


@app.route("/infer", methods=["POST"])
def infer():
    data = request.get_json()
    print("Request data:\n", data)

    # ply_file = "data/interactive_dataset/scene_01_ScanNet_0355_00/label.ply"
    # ply_data = read_ply(ply_file)
    # print(ply_data)

    # click_idx = {
    #     '0': [9477, 8654, 8974, 20551, 21887, 21707],
    #     '1': [10323], 
    #     '2': [4067], 
    #     '3': [17564], 
    #     '4': [18457], 
    #     '5': [9914],
    #     '6': [17457], 
    #     '7': [10914]
    # }
    # click_time_idx = {
    #     '0': [5], #, 6, 7, 8, 9, 10],
    #     '1': [0],
    #     '2': [1],
    #     '3': [2],
    #     '4': [3],
    #     '5': [4],
    #     '6': [6], 
    #     '7': [7]
    # }

    # click_positions = {
    #     k: model.coords[v].tolist() for k, v in click_idx.items()
    # }
    # num_clicks = len(click_idx)

    model.load_scene(data["scene_id"])
    click_time_idx = data["time"]
    click_positions = data["coordinates"]

    click_idx = defaultdict(list)
    for k, v in click_positions.items():
        for i in range(0, len(v), 3):
            click_idx[k].append(find_nearest(model.raw_coords_qv, v[i:i+3]))

    num_clicks = len(click_idx)

    model.get_next_click(
        click_idx=click_idx,
        click_time_idx=click_time_idx,
        click_positions=click_positions,
        num_clicks=num_clicks,
        run_model=True,
        gt_labels=None,
        ori_coords=model.coords,
        scene_name=model.scene_name,
    )
    print(model.object_mask[:,0].max())

    return json.dumps({"label": model.object_mask[:,0].astype(np.int32).tolist()})


if __name__ == "__main__":
    app.run(host="localhost", port=5000, debug=True)
