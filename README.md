# Deep Learning-Based Pneumonia Detection Using Chest X-Ray Images

This project implements a deep learning model for pneumonia detection using chest X-ray images. The model is built on a modified ResNet-18 convolutional neural network architecture [1] and incorporates several training improvements inspired by the techniques described in Bag of Tricks for Image Classification with Convolutional Neural Networks [2].

The system is trained using the Chest X-Ray Images (Pneumonia) dataset [3] and is designed to classify chest X-ray images into pneumonia or normal categories.

The code for this project was ran with Google Colab under a L4 GPU for efficient computation and results.

# Model Architecture

The model uses a ResNet-18 backbone with modifications aimed at improving training efficiency and classification performance. These modifications are based on optimization techniques presented in the Bag of Tricks paper, including improvements to training procedures and regularization methods commonly used in modern convolutional neural network pipelines.

# Dataset

Training and evaluation are performed using the following dataset:

Chest X-Ray Images (Pneumonia)
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

The dataset contains labeled chest X-ray images categorized as:

Normal

Pneumonia

To replicate the results of this code, the dataset must be downloaded to your local machine, saved according to the below directory structure.

# Directory Structure

The project assumes the dataset is stored in the following directory:

/content/chest_xray

All generated outputs, such as trained models, logs, and intermediate files, are saved to:

/content

# train1.py

This script trains a convolutional neural network for pneumonia classification using chest X-ray images. The model is based on the ResNet-18 architecture and is trained **from scratch**, meaning that all network parameters are randomly initialized and learned directly from the training data. The training pipeline includes standard deep learning practices such as data augmentation, regularization, and optimization to learn discriminative features for distinguishing between normal and pneumonia cases.

# train2.py

This script implements a **transfer learning** approach for pneumonia classification. It loads a ResNet-18 model pre-trained on the ImageNet dataset and fine-tunes the network on the chest X-ray dataset. By leveraging representations learned from large-scale natural image data, the model can adapt these features to the medical imaging domain, often leading to improved convergence and classification performance compared to training from scratch.

# References

[1] ResNet-18 PyTorch Implementation
samcw on GitHub
https://github.com/samcw/ResNet18-Pytorch/blob/master/ResNet18.ipynb

[2] He, Tong, et al.
Bag of Tricks for Image Classification with Convolutional Neural Networks.
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019.

Code Implementation:
https://github.com/weiaicunzai/Bag_of_Tricks_for_Image_Classification_with_Convolutional_Neural_Networks

[3] Chest X-Ray Images (Pneumonia) Dataset
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
