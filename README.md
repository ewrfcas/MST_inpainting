# Learning a Sketch Tensor Space for Image Inpainting of Man-made Scenes

[Chenjie Cao](https://github.com/ewrfcas),
[Yanwei Fu](http://yanweifu.github.io/)

![teaser](assets/teaser_new.png)
[arXiv](https://arxiv.org/abs/2103.15087) | [Project Page](https://ewrfcas.github.io/MST_inpainting/)


## Overview
![teaser](assets/overview_new1.png)
We learn an encoder-decoder model, which encodes a Sketch Tensor (ST) space consisted of refined lines and edges. 
Then the model recover the masked images by the ST space. 

### News
- [x] Release the inference codes.
- [ ] Release the GUI codes.
- [ ] Training codes.

### Preparation
1. Preparing the environment. 
2. Download the pretrained masked wireframe detection model [LSM-HAWP](https://drive.google.com/drive/folders/1yg4Nc20D34sON0Ni_IOezjJCFHXKGWUW?usp=sharing) (retrained from [HAWP CVPR2020](https://github.com/cherubicXN/hawp))
3. Download weights for different requires to the 'check_points' fold. 
   [P2M](https://drive.google.com/drive/folders/1uQAzfYvRIAE-aSpYRJbJo-2vBiwit0TK?usp=sharing) (Man-made Places2), 
   [P2C](https://drive.google.com/drive/folders/1td0SNBdSdzMdj4Ei_GnMmglFYOgwUcM0?usp=sharing) (Comprehensive Places2), 
   [shanghaitech](https://drive.google.com/drive/folders/1VsHSRGBpGWjTP-LLZPrtW-DQan3FRnEl?usp=sharing) ([Shanghaitech](https://github.com/huangkuns/wireframe) with all man-made scenes).
   
### Test for a single image
```
python test_single.py --gpu_id 0 --PATH ./check_points/MST_P2C --image_path <your image path> --mask_path <your mask path, 0 means valid and 255 means masked>
```

## Object Removal Examples
![Object removal video](assets/video.gif)

## Comparisons
![ST](assets/shanghaitech_comparisons.png)
![Places2](assets/places2_comparisons.png)