# hoi-attention
This repository show how to extract hand activity information on objects based on visual attention in the task-specific reach-to-grasp cycle. We used perception-based egocentric vision to observe hand-object interactions (HOI) in grasping tasks. Our approach combines object detection with hand skeletal model estimation and visual attention to validate HOI detection. We choose a multilayer Gated Recurrent Unit (GRU) based on Recurrent Neural Networks (RNN) architecture to classify the four main activities when the hand interacts with an object (wonder-reach-grasp-release). We evaluated the algorithm quantitatively on the new dataset we introduced for cup grasping activity. This method can validate the HOI detection with 97.0% precision with less training time for a small data. 
