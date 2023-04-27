# Resolving Datashift in Crowd Counting

The model.py and and prepare_dataset.py are from the github https://github.com/TencentYoutuResearch/CrowdCounting-SASNet/tree/main which is the
Official implementation in PyTorch of SASNet as described in "To Choose or to Fuse? Scale Selection for Crowd Counting" by Qingyu Song *, Changan Wang *, Yabiao Wang, Ying Tai, Chengjie Wang, Jilin Li, Jian Wu, Jiayi Ma.

ShanghaiTech dataset from [GoogleDrive](https://drive.google.com/drive/folders/17WobgYjekLTq3QIRW3wPyNByq9NJTmZ9?usp=sharing)

Generating the density maps for the data:
```
python prepare_dataset.py --data_path ./datas/part_A_final
python prepare_dataset.py --data_path ./datas/part_B_final
```

Run the following command to train the model:
```
python3 train.py --data_path [data path] 
```
Run the following command to do transfer learning on the model:
```
python3 transfer.py --data_path [data path] --model_path [model path]
```

Run the following command to do inference:
```
python3 test.py --data_path [data path] --model_path [model path]
```
