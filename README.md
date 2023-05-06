# Lung Disease Classification
Chest X-rays scans are among the most accessible ways to diagnose lung diseases. This study tries to compare the detection of lung diseases using these scans from  three different datasets using deep neural networks. Three different backbone architectures, ResNet34, MobileNet V3 Large and EfficientNet B1 were used along with a set of models trained using transfer learning. It is observed that MobileNet takes the least amount of time to train while ResNet converges the fastest. Also, EfficientNet performs the best most of the times on Chest X-ray scans. An F1 score of 0.8 for the pneumonia dataset was obtained, 0.98 for the COVID-19 dataset and 0.46 for the multilabel chest X-ray 8 dataset. Finally, models are visualized using t-SNE and gradCAM to understand the features learned by the models and correlate them with the actual effect of the diseases on the lungs.

## Directory Structure

```
├── documentation/              <- All project related documentation and reports
├── deliverables/               <- All project deliverables
├── data/                       <- All project related data files
├── latex/                      <- All latex files for the project
├── models/                     <- Trained models, predictions, and summaries
├── notebooks/                  <- Jupyter notebooks
│  ├── Pneumonia/               <- Notebooks for the pneumonia dataset
│  ├── covid-pneumonia/         <- Notebooks for the COVID dataset
│  ├── xray8/                   <- Notebooks for the Chest X-ray 8 dataset
│  ├── Grad-CAM/                <- Notebooks to generate gradCAM plots
│  ├── t-SNE/                   <- Notebooks to generate t-SNE plots
├── src/                        <- Source code for the project
│  ├── multilabel/              <- Scripts for the multilabel dataset
│  ├── __init__.py              <- Makes src a Python module
├── .gitignore                  <- List of files and folders git should ignore
├── LICENSE                     <- Project's License
├── README.md                   <- The top-level README for developers using this project
├── environment.yml             <- Conda environment file
└── requirements.txt            <- The requirements file for reproducing the environment
```

## Creating the environment
Load conda environment as:
```
conda env create -f environment.yml
```
Install torch in conda environment:
```
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```
## Dataset
We have choosen 3 types of chest X-Ray datasets (Tab. 1) that have varying disease types to ensure that our models are robust. The
main concern while selecting the datasets was the number of images per class as most datasets were highly skewed. We rejected datasets where the images were compressed and noisy as this can lead to mis-diagnosis. This will help reduce the time spent in the pre-processing stage. The dataset links are as follows:
- Dataset 1 - https://www.kaggle.com/datasets/tolgadincer/labeled-chest-xray-images
- Dataset 2 - https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
- Dataset 3 - https://www.kaggle.com/datasets/nih-chest-xrays/data

![sample-dataset](/figures/sample-dataset.png)

- A sample of the dataset can be downloaded from the following [link](https://drive.google.com/file/d1OpvSkIDzOlJUSCLJ-wfXSVsBbfhntun4/view?usp=sharing)

| Dataset                 | No. of Images                | Classes | Image Size  |
|-------------------------|------------------------------|---------|-------------|
| Dataset 1 (COVID)       | `10k:3.6k:1.3k`              | `3`     | `299x299`   |
| Dataset 2 (PNEUMONIA)   | `3k:1.5k:1.5`                | `3`     | `224x224`   |
| Dataset 3 (Chest Xray8) | `25k:12k:6k:5k:3k:2.7k:2.6k` | `7`     | `1024x1024` | 

## Methodology
In this study, 12 models, four for each of the three datasets will be trained. The first three models will be trained from scratch and the fourth model will be trained using transfer learning. The hyperparameters will be fixed across models to produce comparable results. Next, hyperparameters will be tuned to find the best model. Finally, the models will be visualized using t-SNE and Grad-CAM to explain model results. Before training, the images were analysed to come up with pre-processing techniques such as **Histogram Equalization** and **Gaussian Blur** with a 5x5 kernel as [Giełczyk et al.](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0265949) showed that this improved the F1 score by 4% for chest X-ray classification. Visually, the contrast of the scan improved and allowed irregularities to stand out as shown in the figure below.

![histogram](/figures/histogram-equilization.png)

## Training and validating the models
To train and validate the models
1. Create the conda environment
2. Copy the data to the data folder
3. Run the relevant script from the notebooks folder. For example to train the EfficientNet B1 model for the Chest X-ray 8 dataset, run the notebooks/xray8/Multilabel_Training_efficient_net_b1_100_epoch_32_batch.ipynb notebook.

## Running the pre-trained model on the sample dataset
To train and validate the models
1. Create the conda environment
2. Copy the sample data to the data folder
3. Run the relevant script from the notebooks folder. For example to train the transfer learning EfficientNet B1 model for the Chest X-ray 8 dataset, run the notebooks/xray8/Multilabel_Training_efficient_net_b1_100_epoch_32_batch_transfer.ipynb notebook.

## Source code package in PyTorch
Not required. ALl dependencies are present in the environment file.

