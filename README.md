# DLPromoter-SF:A Deep Learning-Based Approach of Predicting Saccharomyces cerevisiae Promoter Strength by Integrating Biological Statistical Features

## Overview

```DLPromoter-SF``` is a deep learning-based approach integrating biological statistical features, which can predict the promoter strength of Saccharomyces cerevisiae according to the input base sequences，and can be used for promoter screening and design in microbial cells such as Saccharomyces cerevisiae, which can accelerate designs of microbial cell factories and contribute to the development of intelligent biomanufacturing.

```DLPromoter-SF``` consists of three modules: the Sequence Feature Extraction Module,  Statistical Feature Extraction Module,  Feature Fusion and Output Module.The Sequence Feature Extraction module leverages a multi-scale convolutional neural network and a Transformer encoder combined with SE attention mechanisms to extract local and global features of promoter sequences. The Statistical Feature Extraction module leverages a feedforward neural network and a gating mechanism are utilized to process four statistical features of promoter sequences, including the 3-mer frequency, the local GC content, the global GC content, and the longest A/T homopolymer length. Finally, the Fusion and Output module uses the FiLM mechanism and a multi-layer perceptron, it achieves feature fusion and outputs predicted strength.

The overall framework of ```DLPromoter-SF``` is shown in the following figure.

![DLPromoter-SF Model Architecture](https://github.com/wuwuwu12310/DLPromoter-SF/blob/main/Model_Architecture.png)


## Description

The project includes the following core files and directory structure:
- The folder `dataset` contains the directory for storing the raw reaction data utilized in the training, testing DLPromoter-SF.
- The file `model.py` contains the the codes of the model for DLPromoter-SF.
- The file `train.py` contains the codes for training the DLPromoter-SF model.
- The file `test.py` contains the codes for testing the DLPromoter-SF model.
- The file `trained_model.pth` contains the script for the trained model pth of DLPromoter-SF.
- The file `extra_norm.json` and `run_config.json` contains the trained configurations and hyperparameters for DLPromoter-SF.

## System Requirements

The proposed ```DLPromoter-SF``` has been implemented, trained, and tested by using `Python 3.8` and `PyTorch 2.4.1` with `CUDA 12.1` and an `NVIDIA RTX4090` graphics card.

The package depends on the Python scientific stack:
```
Python： 3.8.20
torch： 2.4.1
Transformers：  4.46.3 
scikit-learn ： 1.3.2
pandas:  1.5.0 
numpy : 1.23.5 
tqdm:  4.46.1
scipy:  1.9.3  
```

## Usage

### Datasets

- The folder `dataset` contains of 162,982 80-bp Saccharomyces cerevisiae promoter sequences and their respective strength,which are available [here](https://github.com/RK2627/PromoDGDE/tree/main/Data/SC).
-  The folder `dataset` also contains three data subsets, which are divided based on similarity rankings according to the dataset of 162,982 80-bp Saccharomyces cerevisiae promoter sequences, namely `high_similarity` , `medium_similarity` and `low_similarity`。
  Three subsets contains Saccharomyces cerevisiae promoter sequences and their respective strength together,which are available [here](https://github.com/1edv/evolution/blob/master/manuscript_code/model/reproduce_test_data_performance).



### Model Training
We define the model for the **DLPromoter-SF** method in the file `model.py`, where:
- The sequence feature extraction module is implemented based on the multi-scale Convolutional Neural Network (CNN) integrated with Squeeze-and-Excitation (SE) attention mechanisms and Transformer encoders with Squeeze-and-Excitation (SE) attention mechanisms.
- The statistical feature extraction module is implemented based on the FeedForward networks (FFN) and a gating mechanism to process statistical data.

We can run the script of the file `train.py` to train DLPromoter-SF and get the results of  new `trained_model.pth`,`extra_norm.json` and `run_config.json` for testing.

### Model Testing
We can run `test.py` using the new `trained_model.pth`,`extra_norm.json` and `run_config.json` which are obtained from the `train.py` to test and evaulate DLPromoter-SF.



