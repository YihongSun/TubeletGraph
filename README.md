# Tracking and Understanding Object Transformations
### [Project Page](https://tubelet-graph.github.io/) | [Paper](https://arxiv.org/abs/2511.04678) | [Video](https://youtu.be/FOs0BEd5-NY) | [Data Repo](https://github.com/YihongSun/TubeletGraph/tree/main/VOST-TAS#readme) | [🤗 HF Dataset](https://huggingface.co/datasets/yihongs/VOST-TAS)

Official PyTorch implementation for the NeurIPS 2025 paper: "Tracking and Understanding Object Transformations".

<a href="#license"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue.svg"/></a>  

![](assets/teaser.png)

## ⚙️ Installation
The code is tested with `python=3.10`, `torch==2.7.0+cu126` and `torchvision==0.22.0+cu126` on a RTX A6000 GPU.
```bash
# Clone and setup environment
git clone --recurse-submodules https://github.com/YihongSun/TubeletGraph/
cd TubeletGraph/
conda create -n tubeletgraph python=3.10 -y
conda activate tubeletgraph
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
bash thirdparty/setup_ckpts.sh

# Install SAM2 with multi-mask predictions
cd thirdparty/sam2
pip install -e .
pip install -e ".[notebooks]"
python setup.py build_ext --inplace
cd ../..

# Install CropFormer
cd thirdparty
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2 --no-build-isolation
ln -s "$(pwd)"/Entity/Entityv2/CropFormer detectron2/projects/CropFormer
cd detectron2/projects/CropFormer/mask2former/modeling/pixel_decoder/ops
bash make.sh
cd ../../../../../../../..
# conda install -c conda-forge libstdcxx-ng # if libstdc++ version mismatch occurs

# Install FC-CLIP
cd thirdparty/fc-clip
pip install -r requirements.txt
cd ../..
```
In addition, please configure your OpenAI API key (required for GPT-4.1 querying) as follows (add to `~/.bashrc` to persist across sessions).
```bash
export OPENAI_API_KEY="sk-..."
```


## 🔮 Quick Start

You can quickly run TubeletGraph on your own videos using `quick_run.py`:

```bash
python quick_run.py \
    --input_dir <VIDEO_FRAME_DIR> \
    --input_mask <FIRST_FRAME_MASK.png> \
    [--fps 30] \
```
It generates video visualizations (.mp4) and state graph diagrams (.pdf) for all prompt objects in `<FIRST_FRAME_MASK.png>`. 
- `--input_dir`: Directory containing video frames as individual images (e.g., `0001.jpg, 0002.jpg, ...`)
- `--input_mask`: PNG annotation of the first frame with object IDs as pixel values (0=background, 1=object1, 2=object2, ..., 255=ignore)
- `--fps` (optional): Frames per second, default=30

**Example: 0334_cut_fruit_1**
```bash
python quick_run.py --input_dir assets/example/0334_cut_fruit_1 --input_mask assets/example/0334_cut_fruit_1_0000000.png
```
The output visualizations are found under `./_pred_out/predictions/custom-0334_cut_fruit_1-Ours_gpt-4.1`.
- To ensure consistency, the expected outputs are pre-computed and found under `./assets/expected_output/` 


## 📊 Evaluations
### VOST

Please first download [VOST](https://www.vostdataset.org/) and update the corresponding paths in [configs/default.yaml](configs/default.yaml).

🔹 To compute dataset-wise predictions, please run the following lines. 
```bash
python TubeletGraph/run.py -c configs/default.yaml -d vost -s val -m Ours [--gpus 0 1 2 3]  # optional --gpus flag for multi-GPU 
```

🔹 To evaluate tracking / state graph performances, please run the following lines. 
```bash
python3 eval/eval_tracking.py -c configs/default.yaml -p vost-val-Ours
python3 eval/eval_state_graph.py -c configs/default.yaml -p vost-val-Ours_gpt-4.1
```
| Data-Split-Method | $J$ | $J^S$ | $J^M$ | $J^L$ | $P$ | $R$ | $J_{tr}$ | $J_{tr}^S$ | $J_{tr}^M$ | $J_{tr}^L$ | $P_{tr}$ | $R_{tr}$ |
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|vost-val-Ours| 50.9 | 41.3 | 53.0 | 68.6 | 68.1 | 63.7 | 36.7 | 23.6 | 40.2 | 60.1 | 55.2 | 47.0 |

| Data-Split-Method_VLM | Sem-Acc Verb $S_V$ | Sem-Acc Obj $S_O$ | Temp-Loc Pre $T_P$ | Temp-Loc Rec $T_R$ | Overall-Rec(ST) $H_{ST}$ | Overall-Rec $H$ |
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|vost-val-Ours_gpt-4.1| 81.8 (*) | 72.3 (*) | 43.1 | 20.4 | 12.0 | 6.5 (*) |

(*) Minor variance may be observed across runs due to non-deterministic LLM behavior in metric computation.


### VSCOS

Please first download [VSCOS](https://venom12138.github.io/VSCOS.github.io/) and update the corresponding paths in [configs/default.yaml](configs/default.yaml).

🔹 To compute dataset-wise predictions, please run the following lines. 
```bash
python TubeletGraph/run.py -c configs/default.yaml -d vscos -s val -m Ours [--gpus 0 1 2 3]  # optional --gpus flag for multi-GPU 
```

🔹 To evaluate tracking performances, please run the following lines. 
```bash
python3 eval/eval_tracking.py -c configs/default.yaml -p vscos-val-Ours
```
| Data-Split-Method | $J$ | $J^S$ | $J^M$ | $J^L$ | $P$ | $R$ | $J_{tr}$ | $J_{tr}^S$ | $J_{tr}^M$ | $J_{tr}^L$ | $P_{tr}$ | $R_{tr}$ |
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|vscos-val-Ours| 75.9 | 67.8 | 79.1 | 81.0 | 89.3 | 82.9 | 72.2 | 60.7 | 77.6 | 78.4 | 87.4 | 81.7 |


### M<sup>3</sup>-VOS

Please first download [M<sup>3</sup>-VOS](https://zixuan-chen.github.io/M-cube-VOS.github.io/) and update the corresponding paths in [configs/default.yaml](configs/default.yaml).

🔹 To compute dataset-wise predictions, please run the following lines. 
```bash
python TubeletGraph/run.py -c configs/default.yaml -d m3vos -s val -m Ours [--gpus 0 1 2 3]  # optional --gpus flag for multi-GPU 
```

🔹 To evaluate tracking performances, please run the following lines. 
```bash
python3 eval/eval_tracking.py -c configs/default.yaml -p m3vos-val-Ours
```
| Data-Split-Method | $J$ | $J^S$ | $J^M$ | $J^L$ | $P$ | $R$ | $J_{tr}$ | $J_{tr}^S$ | $J_{tr}^M$ | $J_{tr}^L$ | $P_{tr}$ | $R_{tr}$ |
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|m3vos-val-Ours| 74.1 | 67.4 | 78.7 | 78.2 | 88.4 | 79.8 | 64.1 | 55.9 | 68.5 | 70.3 | 82.4 | 71.5|

### Custom Dataset

**Option 1: Quick Run (Recommended for Single Videos)**
```bash
python quick_run.py --input_dir  --input_mask 
```
This automatically handles dataset setup and config generation.

**Option 2: Manual Config (For Dataset-Wide Evaluation)**

🔹 To run on entire datasets with multiple videos, please add the following lines to `configs/default.yaml` under `datasets`:
```
datasets:
  <data_name>:
    name: <data_name>
    data_dir: <DATA_PATH>
    image_dir: <DATA_PATH>/JPEGImages
    anno_dir: <DATA_PATH>/Annotations
    split_dir: <DATA_SPLIT_PATH>
    image_format: <IMAGE_FORMAT>  # e.g., "*.jpg"
    anno_format: <ANNO_FORMAT>  # e.g., "*.png"
    fps: <FPS>  # for visualization fps
```
🔹 Then, run the following line to compute dataset-wise predictions.
```bash
python TubeletGraph/run.py -c configs/default.yaml -d <data_name> -s <split> -m Ours [--gpus 0 1 2 3]  # optional --gpus flag for multi-GPU 
```
- Note that `<DATA_SPLIT_PATH>/<split>.txt` should be found.

🔹 To evaluate tracking performances, please run the following lines.
```bash
python3 eval/eval_tracking.py -c configs/default.yaml -p <data_name>-<split>-Ours
```
- In this case, `<split>` should be `val` and `val-{S/M/L}.txt` should also be found under `<DATA_SPLIT_PATH>`. 

## 🖼️ Visualizations

🔹 To visualizing model predictions, please run the following lines. 
```bash
python3 eval/vis.py -c <CONFIG> -p <PRED> [-i <INSTANCE>_<OBJ_ID>]  # optional -i flag to visualize only 1 instance 
## example
python3 eval/vis.py -c configs/default.yaml -p vost-val-Ours_gpt-4.1  ## visualize all
python3 eval/vis.py -c configs/default.yaml -p vost-val-Ours_gpt-4.1 -i 555_tear_aluminium_foil_1
```
- Output visualization can be found in `_vis_out/predictions/vost-val-Ours_gpt-4.1/`, where `<INSTANCE>_<OBJ_ID>.mp4` and `<INSTANCE>_<OBJ_ID>.pdf` contain the visualized object tracks and state graph, respectively.

🔹 To visualize the internally-computed spatiotemporal partition (tubelets), please run the following lines. 
```bash
python3 TubeletGraph/vis/tubelets.py -c <CONFIG> -d <DATASET> -m <MODEL> -i <INSTANCE>_<OBJ_ID>
## example
python3 TubeletGraph/vis/tubelets.py -c configs/default.yaml -d vost -m cropformer -i 555_tear_aluminium_foil_1
```
- Output visualization can be found at `_vis_out/tubelets/tubelets_vost_cropformer_555_tear_aluminium_foil_1.mp4` showing *input video (top-left)*, *entity segmentation (top-right)*, *initial tubelets (bottom-left)*, and *newly emergent tubelets (bottom-right)*.


## Citation
If you find our work useful in your research, please consider citing our paper:
```bibtex
@article{sun2025tracking,
  title={Tracking and Understanding Object Transformations},
  author={Sun, Yihong and Yang, Xinyu and Sun, Jennifer J and Hariharan, Bharath},
  journal={Advances in Neural Information Processing Systems},
  year={2025}
}
```
