# 11-785-20F-project
Variational Recurrent Autoencoders with Attention, which uses time series data to train an unsupervised model.

Improve the performance of the model by adding attention mechanism and replacing Bi-LSTM with Transformer. Finally, we will implement a multi-class framework based on our previous model to discriminate between different anomalies using the latent features in the z-space.

## Dataset 
[EMHIRES](https://setis.ec.europa.eu/EMHIRES-datasets)
EMHIRES is the first publically available European solar power generation dataset derived from meteorological sources that is available up to NUTS-2 level. We will use it to train our final network.

[Mnist](https://www.kaggle.com/oddrationale/mnist-in-csv)
The MNIST database (Modified National Institute of Standards and Technology database) is a large database of handwritten digits that is commonly used for training various image processing systems. To test the performance of the baseline models and save time, we create abnormal samples in the Mnist dataset and use it to train our baseline networks.

[EGG](https://archive.physionet.org/physiobank/database/adfecgdb/)
The original dataset for "ECG5000" is a 20-hour long ECG downloaded from Physionet. It contains 5,000 Time Series examples (obtained with ECG) with 140 timesteps. Each sequence corresponds to a single heartbeat from a single patient with congestive heart failure. In this project, we selecte part of the normal data as training set ( about 3000 data point) to train the model and then test the model with 300 randomly selected data points.

## Model Description 
[Attention-based LSTM]
This is an auto-encoder model, the basic architecture is a two-layer LSTM encoder and two-layer LSTM decoder. We add attention layer before the LSTM encoder. 

[Transformer Model]
This is a transformer model, the basic architecture is 
