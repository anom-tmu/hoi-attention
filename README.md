# Hand-Object Interaction Detection based on Visual Attention for Independent Rehabilitation Support

![alt text](https://github.com/anom-tmu/hoi-attention/blob/main/hoi_attention_to_knowledge_graph.gif)

This repository show how to extract hand activity information on objects based on visual attention in the task-specific reach-to-grasp cycle. We used perception-based egocentric vision to observe hand-object interactions (HOI) in grasping tasks. Our approach combines object detection with hand skeletal model estimation and visual attention to validate HOI detection. We choose a multilayer Gated Recurrent Unit (GRU) based on Recurrent Neural Networks (RNN) architecture to classify the four main activities when the hand interacts with an object (wonder-reach-grasp-release). We evaluated the algorithm quantitatively on the new dataset we introduced for cup grasping activity. This method can validate the HOI detection with 97.0% precision with less training time for a small data.

![alt text](https://github.com/anom-tmu/hoi-attention/blob/main/01.%20data_acquisition.jpg)

The code is documented and designed to be easy to extend. If you use it in your research, please consider citing this repository (bibtex below). This dataset was created from Kubota Laboratory, Tokyo Metropolitan University and available for academic use. 

## Requirements
This system works with this packages:
  - python 3.8.5
  - pytorch 1.9.0
  - opencv-python 4.5.2.54 
  - numpy 1.19.1
  - pandas 1.1.3
  - scipy 1.5.2
  - other common packages listed in source code.

## Contributing
Contributions to this repository are welcome. Examples of things you can contribute:
  - Speed Improvements.
  - Training on other datasets.
  - Accuracy Improvements.
  - Visualizations and examples.
  - Join our team and help us build even more projects.

## Citation
Use this bibtex to cite this repository: 
```
This research not yet published
```

