# 11-785-20F-project
Variational Recurrent Autoencoders with Attention, which uses time series data to train an unsupervised model.

Improve the performance of the model by replacing Bi-LSTM with Transformer. Finally, we will implement a multi-class framework based on our previous model to discriminate between different anomalies using the latent features in the z-space.

## Dataset 
[EMHIRES](https://setis.ec.europa.eu/EMHIRES-datasets)
EMHIRES is the first publically available European solar power generation dataset derived from meteorological sources that is available up to NUTS-2 level. We will use it to train our final network.

[Mnist](https://www.kaggle.com/oddrationale/mnist-in-csv)
The MNIST database (Modified National Institute of Standards and Technology database) is a large database of handwritten digits that is commonly used for training various image processing systems. To test the performance of the baseline models and save time, we create abnormal samples in the Mnist dataset and use it to train our baseline networks.
