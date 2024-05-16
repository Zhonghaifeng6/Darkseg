# requirements

```
einops	         0.6.1	
numpy	            1.23.0
pip	            23.2.1
scikit-image	   0.21.0	
scikit-learn	   1.3.2	
scikit-posthocs   0.8.0
scipy	            1.10.1
tensorboard	      2.14.0
torch	            1.13.1+cu116	
torchsummary	   1.5.1	
torchvision	      0.14.1+cu116
tqdm	            4.66.1
```
# datasets
```bash
data
   ├─dark（dark images)
   ├─images (infrared images)
   └─masks (labels)
```
# train
```
train_D.py : use dark and infrareds images to  train the network.
train_dark.py : only use dark images to  train the network.
train_infrared.py : only use infrared images to  train the network.
predict_Multi.py : using for multi-classes predict
predict_Single.py : using for single-classes predict
```
