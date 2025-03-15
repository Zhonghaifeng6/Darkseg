## 💥 Motivation
In this work, we found that existing traditional scene perception methods are ineffective for garment perception in low-light environments.
Our DarkSeg model learns illumination-invariant structural representations from infrared images, enabling accurate detection and classification of garment and facilitating robotic grasping in low-light environments.
<p align="center">
<img src="./docs/pipline.png" width=85% height=85% class="center">
</p>

## 💥 Requirements

```
einops 	0.6.1	
numpy	  1.23.0
pip	 23.2.1
scikit-image	0.21.0	
scikit-learn	1.3.2	
scikit-posthocs	0.8.0
scipy	    1.10.1
tensorboard	    2.14.0
tensorboard-data-server   0.7.1	
torch	   1.13.1+cu116	
torchsummary    1.5.1	
torchvision	   0.14.1+cu116
tqdm   4.66.1
```
## 💥 Datasets
The preview of our Darkclothes dataset is as follows.
<p align="center">
<img src="./docs/dataset.png" width=85% height=85% class="center">
</p>

Our Darkclothes is available at [Google Drive](https://drive.google.com/file/d/1Ee8ReaP48tECyilH6Yc08N6aU_qibGch/view?usp=sharing) 
```bash
data
   ├─dark（dark images)
   ├─images (igfrared images)
   └─masks (labels)
   
```
## 💥 Train
```
train.py : use dark and infrareds images to  train the teacher model and student model.
inference_color.py : use for multi-classes predict
inference.py : use for single-classes predict
```

## Acknowlegement
_**DarkSeg**_ is built upon [SegFormer](https://github.com/NVlabs/SegFormer). We thank their authors for making the source code publicly available.
