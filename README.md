# KPIs2024_ZhiJianLife_task2
## Congratulations 🎉🎉🎉🥳🥳🥳
We are proud to announce that our team has achieved second and third place in instance segmentation and detection for the MICCAI 2024 KPIs 2024 Task 2. This is a significant milestone that showcases our efforts and achievements in the field of WSI-level diseased glomeruli segmentation.

## Dataset
See the [data](https://sites.google.com/view/kpis2024/data) for training and validation data.
## Directory list
Create a folder by following the file directory:
``` 
    ├── input
    ├── model
          ├── checkpoint
          ├── mmseg
          ├── tools
    ├── output
    ├── process
          ├── inference_result
          ├── ori
          ├── patch
          ├── tissue
``` 
## Training  
We employ ResNet101-UperNet models for training. Here are the key steps in the training process:

1. **OTSU & FN**: Look at the run_post_process function in the [inference_docker.py](https://github.com/ZhiJianLife/KPIs2024_ZhiJianLife_task2/blob/main/model/inference_docker.py) file.
2. **Overlapping patch extraction**: Look at the [make_patch_mask_data.py](https://github.com/ZhiJianLife/KPIs2024_ZhiJianLife_task2/blob/main/model/make_patch_mask_data.py).
3. **Train model**: Look at the [train.py](https://github.com/ZhiJianLife/KPIs2024_ZhiJianLife_task2/blob/main/model/tools/train.py).
4. **Patch aggregation**: Look at the patch_to_wsi function in the [inference_docker.py](https://github.com/ZhiJianLife/KPIs2024_ZhiJianLife_task2/blob/main/model/inference_docker.py) file.

## Inference 
You can directly load the weights of our model directly inference, the weights are linked [here](https://drive.google.com/file/d/1-0JU7UBY2ZIzu6UIYgNS__15dQDOUitp/view?usp=sharing). Put it under the /model/checkpoint/ directory. Note that you must follow the requirements of the input and output paths of the KPIs2024 task2, see [here](https://sites.google.com/view/kpis2024/evaluation).



