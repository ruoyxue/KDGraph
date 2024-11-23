# KDGraph for Road Graph Extraction

The official code for [KDGraph: A Keypoint Detection Method for Road Graph Extraction from Remote Sensing Images](https://dx.doi.org/10.2139/ssrn.4684597).

## A. KDGraph

1. Prepare SpaceNet3 dataset for KDGraph in the **spacenet_transform** directory.

2. Use the following command for training on SpaceNet3.
	```Python
	python main.py --config ./config.yaml --mode train
	```

3. Inference on SpaceNet3.
	```Python
	python main.py --config ./config.yaml --mode test --checkpoint_path ckpt_path
	```

4. Metrics evaluation.
	```
	cd metrics
	. main.bash
	```

## B. SoR dataset

This dataset is collected from 15 cities around the world to evaluate the ability of different road graph extraction methods in handling shadows and occlusions. The resolution is 0.6 m.

**Baidu Netdisk**
Link: https://pan.baidu.com/s/1TtliEMDvaXgg3WGuL3KzLQ 
Code: 7ddc

## C. Online road detector

This is a web application to conduct online road detection task worldwide. The code can be found in https://github.com/ruoyxue/Online-road-detection-application.

<p align="center">
  <img src="./example.gif">
</p>

The demo video of KDGraph: https://www.youtube.com/watch?v=3bk0pOWXV4M&t=21s.
