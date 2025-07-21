# Image Captioning with PyTorch

This project provides a complete pipeline for training and evaluating an image captioning model using PyTorch. The model is built upon a classic Encoder-Decoder architecture, utilizing a pre-trained ResNet-101 as the encoder and an LSTM network as the decoder.

![Project Banner](httpses/Admin/Documents/image_captioning/data/images/1000268201_693b08cb0e.jpg) 
*Caption: a man is performing a trick on a ramp*

---

## Table of Contents
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [How to Use](#how-to-use)
  - [Training](#training)
  - [Generating Captions (Inference)](#generating-captions-inference)
  - [Evaluating the Model](#evaluating-the-model)

---

## Features
- **Encoder-Decoder Framework**: Employs a robust architecture with a CNN encoder and an RNN decoder.
- **Pre-trained Encoder**: Leverages the power of ResNet-101 pre-trained on ImageNet for rich feature extraction.
- **LSTM Decoder**: Uses a Long Short-Term Memory (LSTM) network to generate sequential text descriptions.
- **Multiple Decoding Methods**:
  - **Greedy Search**: A fast, straightforward method for generating captions.
  - **Beam Search**: A more advanced search strategy that explores multiple possibilities to generate higher-quality, more coherent captions.
- **Training and Evaluation Scripts**: Comes with clear, easy-to-use scripts to train the model from scratch, generate predictions, and evaluate performance using BLEU scores.
- **Progress Tracking**: Integrated with `tqdm` for visual feedback during training and evaluation.

---

## Model Architecture

The model consists of two main components:

1.  **Encoder (CNN)**: We use a **ResNet-101** model, pre-trained on the ImageNet dataset. The final fully connected layer is removed, and the output of the last convolutional block is used as the image embedding. This provides a rich, high-level representation of the image's content.

2.  **Decoder (LSTM)**: An **LSTM (Long Short-Term Memory)** network takes the image feature vector from the encoder as its initial input. It then generates the caption word by word, with the output of each step being fed as the input for the next, until an `<EOS>` (end-of-sequence) token is produced.

---

## Dataset
This project is designed for the **Flickr8k dataset**. It contains 8,000 images, each paired with five different human-written captions.

- **Download**: You can download the dataset from Kaggle: [Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- **Setup**: After downloading, unzip the contents and place the `captions.txt` file and the `images` folder inside the `data/` directory.

---

## Project Structure
```
image_captioning/
├── data/
│   ├── captions.txt
│   └── images/
│       ├── 1000268201_693b08cb0e.jpg
│       └── ...
├── models/
│   ├── model.py
│   └── weights/
├── utils/
│   └── data_loader.py
├── runs/
│   └── (Tensorboard logs)
├── test_new_images/
│   └── (Place your test images here)
├── train.py
├── predict.py
├── evaluate.py
├── check_data.py
└── README.md
```

---

## Setup & Installation

Follow these steps to get the project up and running on your local machine.

**1. Clone the Repository**
```bash
git clone <your-repository-link>
cd image_captioning
```

**2. Set up the Dataset**
- Download the Flickr8k dataset from the link provided above.
- Make sure the `data` directory is structured as shown in the [Project Structure](#project-structure) section.

**3. Install Dependencies**
- It's recommended to use a virtual environment.
- Install all required Python packages using the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

---

## How to Use

### Training
To train the model from scratch, simply run the `train.py` script. The trained model weights will be saved in the `models/weights/` directory after each epoch.

```bash
python train.py
```
You can modify hyperparameters like `learning_rate`, `num_epochs`, `embed_size`, etc., directly within the `train.py` file.

### Generating Captions (Inference)
Use the `predict.py` script to generate a caption for any image.

```bash
python predict.py --image <path_to_your_image.jpg> --model_path <path_to_your_model.pth>
```

**Example:**
```bash
# Using an image from the dataset
python predict.py --image data/images/1000268201_693b08cb0e.jpg --model_path models/weights/image_captioning_epoch_50.pth

# Using a new image you downloaded
python predict.py --image test_new_images/my_vacation_photo.jpg --model_path models/weights/image_captioning_epoch_50.pth
```
The script will print the generated caption to the console.

### Evaluating the Model
To evaluate the performance of a trained model on the test set, use the `evaluate.py` script. It will calculate the BLEU scores.

```bash
python evaluate.py --model_path <path_to_your_model.pth>
```

**Example:**
```bash
python evaluate.py --model_path models/weights/image_captioning_epoch_50.pth
```
The script will output the BLEU-1, BLEU-2, BLEU-3, and BLEU-4 scores, giving you a quantitative measure of the model's performance. 