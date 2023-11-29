# GPT2 Implementation from Scratch

This repository contains an implementation of the GPT2 model from scratch.

The implementation is well-commented for easy understanding.

## Task Wise Google Colab Files (No Dependencies Needed)

- Task 1: [Contlo_Task1](https://drive.google.com/file/d/1xBjQFJH3wiqPMZkc0-prBI1ZHAm6XWye/view?usp=sharing)
- Task 2: [Contlo_Task2](https://drive.google.com/file/d/1L00wRlUclajPH-eREL5Pz1cGyAPMxCdS/view?usp=sharing)
- Task 3: [Contlo_Task3](https://drive.google.com/file/d/1fTlXCSvDJ9kU-kFif5P1Rhf5qJHNP9Pv/view?usp=sharing)

## Folder Structure
```
.
├── GPT2Model
│   ├── config.py
│   ├── layers.py
│   └── model.py
├── GPT2Trainer
│   ├── config.py
│   ├── dataset.py
│   └── trainer.py
├── inference.py
├── train.py
├── requirements.txt
└── README.md
```



### Module 1: GPT2Model

This module focuses on building the GPT2 model architecture. Here's a breakdown of the key components:

- **config.py:** Specifies configurations and model parameters, including options for positional embeddings, multi-head attention, or grouped query attention.

- **layers.py:** Implements all the necessary layers for the GPT2 model, such as attention, feed-forward, layer normalization, etc.

- **model.py:** Defines the main class `GPT2`, which takes the configuration and layers to build the GPT2 model.

### Module 2: GPT2Trainer

This module is dedicated to training the GPT2 model. It includes the following components:

- **config.py:** Specifies training configurations, such as the number of epochs, batches, GPU usage, distributed data parallelism (DDP), or fully sharded data parallelism (FSDP).

- **dataset.py:** Defines a custom dataset class for loading custom text data. This class is designed to work seamlessly with the GPT2 model.

- **trainer.py:** Implements the main class `Trainer`, which contains a training loop for training the GPT2 model using the custom dataset and the specified configurations.

## Additional Files

- **inference.py:** This file handles loading the trained model and performs inference on new data.

- **train.py:** This script loads the model and performs training using the `Trainer` class defined in the GPT2Trainer module.

## Note 

Make sure to install the required dependencies listed in the `requirements.txt` file before running the code.


