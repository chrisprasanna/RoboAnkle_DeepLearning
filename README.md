
# RoboAnkle_DeepLearning
Deep learning scripts used to accurately predict human-robot system dynamics for future prosthesis control methods

## Project Goal
Given the increasing amount of data available, the development of advanced machine and deep learning techniques toward robotic prosthesis applications is now possible. The goal of this project was to develop data-driven models to output forward predictions of a prototype prosthetic's ankle torque values.

Modeling these human-robot system dynamics would be useful for understanding the system behavior and planning control actions accordingly. A system model can also allow for model-based control methods to be used, which are much more sample-efficient and converge much quicker when compared to model-free techniques.

## Challenges
Despite their promise, data-driven regression models have not been implemented in robotic prosthesis control and modeling human-prosthesis dynamical behavior remains a challenge due to highly nonlinear behavior. Traditional system identification techniques are limited since they require some knowledge of the system, cannot learn spatial-temporal relationships on their own, and cannot extract the most important features from a dataset.

## Solution
To account for the nonlinear dynamics of the system, deep neural network (DNN) architectures with internal state (memory) gates were trained to better predict sudden dynamic changes and possible disturbances to the system. Additionally, attention mechanisms were investigated in order to enhance and devote more computational power to the small but important subset of the feature space while fading out the rest.


## Features

- Three deep neural network architectures were developed and trained: (i) feedforward network (FFN), (ii) gated recurrent unit (GRU), and (iii) dual-stage attention-based recurrent neural network (DA-RNN)
- Gate mechanisms were added which add information to or remove information from the state across timesteps, making them particularly effective for learning temporal dependencies
- Attention mechanisms were included which adaptively extract the most relevant features and hidden state information using an encoder-decoder architecture
- Machine learning data processing methods such as tensor conversion, shuffling, mini-batches, feature normalization, data augmentation, and rolling lookback time windows were utilized for improved network training
- Network training hyperparameters were optimized using the Optuna framework which utilized Bayesian search techniques and trial pruning
- Implemented validation dataset evaluation and an early stopping protocol in order to reduce the chance of overfitting

  
## Acknowledgements

 - Preliminary data collected by [Jonathan Realmuto](https://github.com/jonreal)
 - [DA-RNN Code](https://github.com/Seanny123/da-rnn)
