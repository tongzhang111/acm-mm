import json
import os
from tqdm import tqdm

# load config
with open("/data/zhangtong1/tubedetr/config/hcstvg.json", "r") as f:
    cfg = json.load(f)
video_path = os.path.join(cfg.get("hcstvg_vid_path"), "video")
ann_path = cfg.get("hcstvg_ann_path")

# get video to path mapping
dirs = os.listdir(video_path)
vid2path = {}
for dir in dirs:
    files = os.listdir(os.path.join(video_path, dir))
    for file in files:
        assert os.path.exists(os.path.join(video_path, dir, file))
        vid2path[file[:-4]] = os.path.join(dir, file)

# preproc annotations
files = ["train.json", "test.json"]
for file in files:
    videos = []
    annotations = json.load(open(os.path.join(ann_path, file), "r"))
    for video, annot in tqdm(annotations.items()):
        out = {
            "original_video_id": video[:-4],
            "frame_count": annot["img_num"],
            "width": annot["width"],
            "height": annot["height"],
            "tube_start_frame": annot["st_frame"],  # starts with 1
            "tube_end_frame": annot["st_frame"] + len(annot["bbox"]),  # excluded
            "tube_start_time": annot["st_time"],
            "tube_end_time": annot["ed_time"],
            "video_path": vid2path[video[:-4]],
            "caption": annot["caption"],
            "video_id": len(videos),
            "trajectory": annot["bbox"],
        }
        videos.append(out)

    json.dump(videos, open(os.path.join(ann_path, file[:-5] + "_proc1.json"), "w"))