# tree_video_caption

Implementation of `Multi-modal Dependency Tree for Video Captioning`

### Requirements:
- Python 3.8
- Java 8 (for coco-caption)
- torch 1.7.0
- fairseq 0.10.1
- (other packages in requirements.txt)

### Usage:
 put the content of coco-caption (https://github.com/tylin/coco-caption) in src/
 ```
 cd src
 python main/train_tree.py train --dataset <dataset> --model <model>
 ```
