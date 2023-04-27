# Resolving Datashift in Crowd Counting
 
## Visualizations for the scale-adaptive selection
The proposed adaptive selection strategy automatically learns the internal relations and the following visualizations demonstrate its effectiveness.

<p align="center"><img src="imgs/fig1.png" width="80%"/>

ShanghaiTech dataset from [GoogleDrive](https://drive.google.com/drive/folders/17WobgYjekLTq3QIRW3wPyNByq9NJTmZ9?usp=sharing)

```
Generating the density maps for the data:
```
python prepare_dataset.py --data_path ./datas/part_A_final
python prepare_dataset.py --data_path ./datas/part_B_final
```

## Training

## Testing
 
##Transfer Learning

## The network
The overall architecture of the proposed SASNet mainly consists of three components: U-shape backbone, confidence branch and density branch.

<img src="imgs/main.png"/>
