# Resolving Datashift in Crowd Counting

The model.py and and prepare_dataset.py are from the github https://github.com/TencentYoutuResearch/CrowdCounting-SASNet/tree/main which is from the paper "To Choose or to Fuse? Scale Selection for Crowd Counting"

ShanghaiTech dataset from [GoogleDrive](https://drive.google.com/drive/folders/17WobgYjekLTq3QIRW3wPyNByq9NJTmZ9?usp=sharing)

Generating the density maps for the data:
```
python prepare_dataset.py --data_path ./datas/part_A_final
python prepare_dataset.py --data_path ./datas/part_B_final
```
