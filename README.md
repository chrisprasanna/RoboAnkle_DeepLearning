
# RoboAnkle_DeepLearning

  

This repository contains software that was used in the deep learning pipeline of the following research:

- [Using Deep Learning Models and Wearable Sensors to Predict Prosthetic Ankle Torques](https://www.techrxiv.org/articles/preprint/Using_Deep_Learning_Models_and_Wearable_Sensors_to_Predict_Prosthetic_Ankle_Torques/19790236)

  

## Description

  

This software was developed to build models that can estimate and predict powered ankle-foot prosthesis (PAFP) torques. Typically, ankle torques are computed offline using inverse dynamics from motion capture. However, this method is time-intensive, limited to a gait laboratory setting, and requires a large array of reflective markers to be attached to the body. A practical alternative must be developed to provide biomechanical information to high-bandwidth prosthesis control systems to enable predictive controllers. This software applies deep learning to build dynamical system models capable of accurately estimating and predicting prosthetic ankle torque from inverse dynamics using only six signals that can be accessed in real time using wearable sensors. This application of deep learning provides an avenue towards the development of predictive control systems for powered limbs aimed at optimizing prosthetic ankle torque (i.e., impedance control).

  

![losses](/img/block_diagram.png)

  

## Table of Contents

- [Installation](#installation)

- [Usage](#usage)

- [Features](#features)

- [Results](#results)

- [Tests](#tests)

- [Future Work](#future-work)

- [Credits](#credits)

- [License](#license)

  

## Installation

1. Download this repository and move it to your desired working directory

2. Open the Anaconda prompt

3. Navigate to your working directory using the cd command

4. Run the following command in the Anaconda prompt:<br/>
&nbsp;&nbsp;&nbsp;&nbsp;  ````conda env create --name NAME --file environment.yml````<br/>

&nbsp;&nbsp;&nbsp;&nbsp;where NAME needs to be changed to the name of the conda virtual environment for this project. This environment contains all the package installations and dependencies for this project.

5. Run the following command in the Anaconda prompt:<br/>
&nbsp;&nbsp;&nbsp;&nbsp;  ````conda activate NAME````

&nbsp;&nbsp;&nbsp;&nbsp; This activates the conda environment containing all the required packages and their versions. 

6. Run the following command in the Anaconda prompt:<br/>
&nbsp;&nbsp;&nbsp;&nbsp;  ````pip install -e .````<br/>

&nbsp;&nbsp;&nbsp;&nbsp;This command installs a project package and tells Python to look for the library code within the /src folder.

  

## Usage

The primary neural network training and evaluation pipeline script is scripts/main_dnn_training_pipeline.py. The main pipeline script calls functions from packages in the src/ folder, including data processing, training protocols, and other computations. Intermediate neural networks during training are saved as .pt files to the results/ folder. In addition, fully-trained neural networks from Optuna trials are saved as pickle files to the results/ folder. The final test results, including test performance metrics and the best-performing hyperparameters in the Optuna hyperparameter optimization, are also saved as a pickle file to this folder. Finally, the test results are saved to a multi-page pdf located in the /results folder. This repository was set up using practices described in the [Good Research Code Handbook](https://goodresearch.dev/index.html#).

  

### Data Structure

Your data structure class should be a dictionary with the following key-value pairs:

- data-dict: each key of this dictionary represents a collected time series (e.g., walking trial) and each value is a Pandas DataFrame where the rows represent time steps (i.e., samples) and each column represents a variable (i.e., measurements or sensor signals). Each column should be labeled and there should be one column that represents time.

- file names-list: each element of this list contains a string of characters describing each collected time series (e.g., walking trial).

### Variables to Change

There are a few Python variables that call out input feature and target output variable names. These should be changed to correspond to the model inputs and outputs for your particular system.   
- **/src/data_processing.py**: within the "get_lookback_window" function, change the "columns" parameters for variables data_x and data_y. 
&nbsp;&nbsp;&nbsp;&nbsp;         
	````
        tmp_x = np.array([omega_motor, hip_position, ankle_position, left_force, right_force, U])
        data_x = pd.DataFrame(np.transpose(tmp_x),
                           columns=['w_m','th_h','th_a','F_l','F_r','i_m'])
        tmp_y = np.array([ankle_torque])
        data_y = pd.DataFrame(np.transpose(tmp_y), columns=['Tau'])
	````
	<br/>

- **/src/testing_and_evaluation.py**: within the "main_test" function, change the "target_signal_names" variable to correspond to the model output(s) for your particular system. 



### Folders

- **data**: Where you put raw data for your project. Unfortunately, we are not able to share our raw data for administrative/privacy reasons, but I still thought it would be cool to share the code.

- **results**: Where you put results, including checkpoints, pickle files, as well as figures and tables.

- **scripts**: Where you put the main executable scripts - Python and bash alike - as well as any .ipynb notebooks.

- **src**: Python modules for the project. This is the kind of python code that you import.

- **tests**: Where you put tests for your code.

  

### Files

- **.gitignore** contains a list of files that git should ignore.

- **README.md** contains a description of the project.

- **environment.yml** allows you to recreate the exact Python environment that is used to run this analysis in a virtual environment.

- **setup.py** allows you to pip-install our custom packages in src/ and import them into the main pipeline script even though they are in a different folder.

## Features

  

- Three neural network model architectures:

- Feedforward network (FFN)

- Gated recurrent unit (GRU)

- Dual-stage attention-based gated recurrent unit (DA-GRU)

- Implementation of a hyperparameter optimization protocol via the Optuna Python library.

- Options for long-term time series forecasting

- Data preprocessing modules (e.g., splitting, resampling, normalization, etc.)

- Neural network training progress bar via the tqdm Python library.

- Early stopping regularization with the validation dataset used to avoid overfitting on the training dataset.

- Flexible with hyperparameter values (e.g., number of hidden units, number of layers, learning rate, etc.) and constant variable values (e.g., number of inputs, number of outputs, number of Optuna trials, etc.)

- Modules for model prediction analysis and visualizations using the test dataset.

  

## Results
Sample results from a complete DNN training, validation, and testing run.

### Training and Validation Loss

<p  align="center">

<img  src="https://github.com/chrisprasanna/RoboAnkle_DeepLearning/blob/main/img/loss_plot.png"  width=75%>

</p>

  
  

### One-Sample-Ahead Predictions

<p  align="center">

<img  src="https://github.com/chrisprasanna/RoboAnkle_DeepLearning/blob/main/img/FFN_Test1_1Ahead.jpg"  width=75%>

<img  src="https://github.com/chrisprasanna/RoboAnkle_DeepLearning/blob/main/img/GRU_Test1_1Ahead.jpg"  width=75%>

<img  src="https://github.com/chrisprasanna/RoboAnkle_DeepLearning/blob/main/img/DA-GRU_Test1_1Ahead.jpg"  width=75%>

</p>

  

### Twenty-Samples-Ahead Predictions

<p  align="center">

<img  src="https://github.com/chrisprasanna/RoboAnkle_DeepLearning/blob/main/img/FFN_Test1_20Ahead.jpg"  width=75%>

<img  src="https://github.com/chrisprasanna/RoboAnkle_DeepLearning/blob/main/img/GRU_Test1_20Ahead.jpg"  width=75%>

<img  src="https://github.com/chrisprasanna/RoboAnkle_DeepLearning/blob/main/img/DA-GRU_Test1_20Ahead.jpg"  width=75%>

</p>

  

### Model Comparisons

  

![Model Comparison](/img/model_comparison.png)

  

One-sample ahead model predictions of powered ankle-foot prosthesis (PAFP) torques across gait cycles compared to motion capture (MoCap) measurements. The periodic time series are time-normalized across the gait cycle for better visualization. The solid lines represent the mean and the width of the traces represent ±1 standard deviation.

  

## Tests

In progress.

  

[comment]:<>  (Go the extra mile and write tests for your application. Then provide examples on how to run them here.)


## Future Work

This deep learning pipeline is currently being expanded to train deep neural network models that characterize multiple input, multiple output (MIMO) robotic prosthesis systems, specifically for the [COBRA system](https://github.com/ajanders/cobra-knee). This is in contrast to the previously developed models that have multiple inputs from wearable sensors and only a single output (i.e., prosthetic ankle torque). Training accurate MIMO system models using deep learning enables us to run forward simulations (i.e., rollouts). In other words, we can simulate the response of the system to arbitrary inputs at various initial states. This tool would allow us to test and tune various prosthesis control methods and configurations prior to implementation. In addition, it also provides a means for experimentation and exploration of control actions without risking the safety of the prosthesis user.
  

## Credits

  

- Patrick J Mineault & The Good Research Code Handbook Community (2021). [The Good Research Code Handbook](https://goodresearch.dev/index.html). Zenodo. [doi:10.5281/zenodo.5796873](doi:10.5281/zenodo.5796873)

- Anthony Anderson. "COBRA-DeepLearning-2.0". (2022). 

- Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta,and Masanori Koyama. 2019. [Optuna: A Next-generation Hyperparameter Optimization Framework](https://arxiv.org/abs/1907.10902). In KDD.

- Yao Qin, Dongjin Song, Haifeng Chen, Wei Cheng, Guofei Jiang, Garrison W. Cottrell. [A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction](https://arxiv.org/pdf/1704.02971.pdf). arXiv preprint arXiv:1704.02971 (2017).

- Alexey Kurochkin. ["DA-RNN"](https://github.com/KurochkinAlexey/DA-RNN.git). (2019).

- Bjarte Mehus Sunde. ["Early Stopping for PyTorch"](https://github.com/Bjarten/early-stopping-pytorch#:~:text=Early%20stopping%20is%20a%20form,a%20row%20the%20training%20stops.). (2019).

  

## License

  

MIT License

  

Copyright (c) 2022 Christopher Prasanna

  

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

  

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

  

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

  

---