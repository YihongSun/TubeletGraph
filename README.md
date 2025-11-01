# Tracking and Understanding Object Transformations
### [Project Page](https://tubelet-graph.github.io/) | [Paper (Coming Soon)](https://github.com/YihongSun/TubeletGraph) | [Video (Coming Soon)](https://github.com/YihongSun/TubeletGraph)

Official PyTorch implementation for the NeurIPS 2025 paper: "Tracking and Understanding Object Transformations".

<a href="#license"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue.svg"/></a>  

![](assets/teaser.png)

## TODOs (By 12/2)
- [x] Expand and polish [VOST-TAS](https://github.com/YihongSun/TubeletGraph/tree/main/VOST-TAS) documentations and visualizations - Done (10/31)
- [ ] Expand and polish main code documentations
- [ ] Add quick demo from input to all predictions


## ⚙️ Installation
The code is tested with `python=3.10`, `torch==2.7.0+cu126` and `torchvision==0.22.0+cu126` on a RTX A6000 GPU.
```
git clone --recurse-submodules https://github.com/YihongSun/TubeletGraph/
cd TubeletGraph/
conda create -n tubeletgraph python=3.10
conda activate tubeletgraph
TODO: add more packages
pip install torch==1.12.1 torchvision==0.13.1
pip install matplotlib opencv-python tqdm scikit-image pycocotools omegaconf
pip install imageio
pip install imageio[ffmpeg]
```
In addition, please install [SAM2 with multi-mask predictions](https://github.com/YihongSun/sam2/tree/fb5e452074cd8bf2da3e2d9b4108e480b7f07276) in [thirdparty](thirdparty) according to official documentations. Finally, please install [CropFormer](https://github.com/qqlu/Entity/blob/6e7e13ac91ef508088e1b848167c01f19b00b512/Entityv2/README.md) and [FC-CLIP](https://github.com/bytedance/fc-clip/tree/2b0bbe213070d44da9182530fa2e826fef03f974) with separate environments and update the paths in [configs/default.yaml](configs/default.yaml), accordingly.


## 🔮 Predictions
Computing entities (region proposals)
```
python3 TubeletGraph/entity_segmentation/cropformer.py -c <CONFIG> -d <DATASET> -s <SPLIT> --num_workers <N> --wid <I>
## example
conda activate cropformer      ## requires separation installation
python3 TubeletGraph/entity_segmentation/cropformer.py -c configs/default.yaml -d vost -s val
```

Computing tubelets 
```
python3 TubeletGraph/tubelet/compute_tubelets_sam.py -c <CONFIG> -d <DATASET> -s <SPLIT> --num_workers <N> --wid <I>
## example
python3 TubeletGraph/tubelet/compute_tubelets_sam.py -c configs/default.yaml -d vost -s val
```

Computing semantic similarity 
```
python3 TubeletGraph/semantic_sim/compute_sim_fcclip.py -c <CONFIG> -d <DATASET> -s <SPLIT> -t <TUBELET_NAME> --num_workers <N> --wid <I>
## example
conda activate fcclip           ## requires separation installation
python3 TubeletGraph/semantic_sim/compute_sim_fcclip.py -c configs/default.yaml -d vost -s val -t tubelets_vost_cropformer
```

Compute predictions
```
python3 TubeletGraph/get_prediction.py -c <CONFIG> -d <DATASET> -s <SPLIT> -m <METHOD>
## example
python3 TubeletGraph/get_prediction.py -c configs/default.yaml -d vost -s val -m Ours
```

Obtain state graph description
```
python3 TubeletGraph/vlm/prompt_vlm.py -c <CONFIG> -p <PRED>
## example
python3 TubeletGraph/vlm/prompt_vlm.py -c configs/default.yaml -p vost-val-Ours
```

## 📊 Evaluations
Compute tracking performances
```
python3 eval/eval.py -c <CONFIG> -p <PRED>
## example
python3 eval/eval.py -c configs/default.yaml -p vost-val-Ours
```

Compute state-graph performances
```
python3 eval/compute_temploc_pr.py -c <CONFIG> -p <PRED>
python3 eval/compute_sem_acc.py -c <CONFIG> -p <PRED>
## example
python3 eval/compute_temploc_pr.py -c configs/default.yaml -p vost-val-Ours_gpt-4.1
python3 eval/compute_sem_acc.py -c configs/default.yaml -p vost-val-Ours_gpt-4.1
```


## 🖼️ Visualizations
Visualizing entity segmentations
```
python3 eval/vis_entities.py -c <CONFIG> -d <DATASET> -m <MODEL> -i <INSTANCE>
## example
python3 eval/vis_entities.py -c configs/default.yaml -d vost -m cropformer -i 3161_peel_banana 
```

Visualizing tubelets
```
python3 eval/vis_tubelets.py -c <CONFIG> -d <DATASET> -m <MODEL> -i <INSTANCE>_<OBJ_ID>
## example
python3 eval/vis_tubelets.py -c configs/default.yaml -d vost -m cropformer -i 3161_peel_banana_1
```

Visualizing state graphs
```
python3 eval/vis_tubelets.py -c <CONFIG> -p <PRED>
## example
python3 eval/vis_states.py -c configs/default.yaml -p vost-val-Ours_gpt-4.1
```
