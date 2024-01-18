
FSM is a new architecture for spatio-temporal video grounding that consists of an efficient video and text encoder that models spatial multi-modal interactions over sparsely sampled frames and a space-time decoder that jointly performs spatio-temporal localization.

This repository provides the code for our paper. This includes:
- Software setup, data downloading and preprocessing instructions for the VidSTG, HC-STVG1 datasets
- Training scripts and pretrained checkpoints
- Evaluation scripts and demo

## Setup
Download [FFMPEG](https://ffmpeg.org/download.html) and add it to the `PATH` environment variable. 
The code was tested with version `ffmpeg-4.2.2-amd64-static`.
Then create a conda environment and install the requirements with the following commands:
```
conda create -n tubedetr_env python=3.8
conda activate tubedetr_env
pip install -r requirements.txt
```

## Data Downloading
Setup the paths where you are going to download videos and annotations in the config json files.

**VidSTG**: Download VidOR videos and annotations from [the VidOR dataset providers](https://xdshang.github.io/docs/vidor.html).
Then download the VidSTG annotations from [the VidSTG dataset providers](https://github.com/Guaranteer/VidSTG-Dataset).
The `vidstg_vid_path` folder should contain a folder `video` containing the unzipped video folders. 
The `vidstg_ann_path` folder should contain both VidOR and VidSTG annotations.

**HC-STVG**: Download HC-STVG1 and HC-STVG2.0 videos and annotations from [the HC-STVG dataset providers](https://github.com/tzhhhh123/HC-STVG).
The `hcstvg_vid_path` folder should contain a folder `video` containing the unzipped video folders. 
The `hcstvg_ann_path` folder should contain both HC-STVG1 and HC-STVG2.0 annotations.

## Data Preprocessing
To preprocess annotation files, run:
```
python preproc/preproc_vidstg.py
python preproc/preproc_hcstvg.py
python preproc/preproc_hcstvgv2.py
```

## Training
Download [pretrained RoBERTa tokenizer and model weights](https://huggingface.co/transformers/v2.6.0/pretrained_models.html) in the `TRANSFORMERS_CACHE` folder.
Download [pretrained ResNet-101 model weights](https://pytorch.org/vision/stable/models.html) in the `TORCH_HOME` folder.
Download [MDETR pretrained model weights](https://github.com/ashkamath/mdetr) with ResNet-101 backbone in the current folder.

**VidSTG** To train on VidSTG, run:
```
python -m torch.distributed.launch --nproc_per_node=NUM_GPUS --use_env main.py --ema \
--load=pretrained_resnet101_checkpoint.pth --combine_datasets=vidstg --combine_datasets_val=vidstg \
--dataset_config config/vidstg.json --output-dir=OUTPUT_DIR
```

**HC-STVG1**
To train on HC-STVG1, run:
```
python -m torch.distributed.launch --nproc_per_node=NUM_GPUS --use_env main.py --ema \
--load=pretrained_resnet101_checkpoint.pth --combine_datasets=hcstvg --combine_datasets_val=hcstvg \
--dataset_config config/hcstvg.json --epochs=40 --eval_skip=40 --output-dir=OUTPUT_DIR
```


## Evaluation
For evaluation only, simply run the same commands as for training with `--resume=CHECKPOINT --eval`. 
For this to be done on the test set, add `--test` (in this case predictions and attention weights are also saved).

