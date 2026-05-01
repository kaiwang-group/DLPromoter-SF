# DLPromoter-SF

## Overview

```DLPromoter-SF``` is a Deep Learning-based approach of predicting _saccharomyces cerevisiae_ Promoter strength by integrating biological Statistical Features.

```DLPromoter-SF``` consists of three modules: the sequence feature extraction module, statistical feature extraction module, and feature fusion and output module. The sequence feature extraction module leverages a multi-scale convolutional neural network and a Transformer encoder combined with SE attention mechanisms to extract local and global features of promoter sequences. The statistical feature Extraction module employs a feedforward neural network and a gating mechanism to process four statistical features of promoter sequences, including the 3-mer frequency, the local GC content, the global GC content, and the longest A/T homopolymer length. The fusion and output module uses a FiLM mechanism and a multi-layer perceptron to achieve feature fusion and output predicted strength values.

The overall framework of ```DLPromoter-SF``` is shown in the following figure.

![DLPromoter-SF Model Architecture](https://github.com/wuwuwu12310/DLPromoter-SF/blob/main/Model_Architecture.png)


## Description
There are totally one folder and six files, which are briefly described as below.
- The folder `dataset` contains the datasets for trainings and testings of ```DLPromoter-SF```.
- The file `model.py` is the code script of ```DLPromoter-SF```.
- The file `train.py` is the code script of training.
- The file `test.py` is the code script of testing.
- The file `trained_model.pth` contains the learned parameters of the trained ```DLPromoter-SF```.
- The file `extra_norm.json` contains the normalization parameters of the trained ```DLPromoter-SF```. 
- The file `run_config.json` contains the configurations and setting values of ```DLPromoter-SF```.

## System Requirements

The proposed DLPromoter-SF has been implemented, trained, and tested by using ```Python 3.8``` and ```PyTorch 2.4.1``` with ```CUDA 12.1``` and an ```NVIDIA RTX4090``` graphics card.

The package depends on the Python scientific stack:
```
Python: 3.8.20
torch: 2.4.1+cu121
Transformers: 4.46.3 
scikit-learn: 1.3.2
pandas: 1.5.0 
numpy: 1.23.5 
tqdm: 4.46.1
scipy: 1.9.3  
```

## Usage

### Datasets

- The folder `dataset` contains the files `wrcprocess_train_data162982.csv` and `wrcprocess_test_data162982.csv`, which are respectively the training and testing subsets of the 162,982 _saccharomyces cerevisiae_ promoters dataset.This dataset is also available in other sources,such as https://github.com/RK2627/PromoDGDE/tree/main/Data/SC.
- The folder `dataset` contains the files `high_similarity.csv`, `medium_similarity.csv`, and `low_similarity.csv`,which are respectively contains 12,708  _saccharomyces cerevisiae_ promoters dataset.This dataset is also available in other sources,such as https://github.com/1edv/evolution/blob/master/manuscript_code/model/reproduce_test_data_performance.



### Model Training
We can run the code script of `train.py` to train ```DLPromoter-SF```, which generates the files `trained_model.pth`, `extra_norm.json`, and `run_config.json`. 

### Model Testing
We can run the file `test.py` to test and evaulate prediction performances of the trained ```DLPromoter-SF```.,where the files `trained_model.pth`, `extra_norm.json`, and `run_config.json` can be automatically loaded.



