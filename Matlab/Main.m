% Requires R2019b or later


close all; 
clear, clc

%% Parameters

% Number of industries
param.numIndustries = 2;

% Number of devices
param.numDevices = 5;

% Number of local iterations
param.localIterations = 1:3;

% Number of global iterations
param.globalIterations = 1:3;

% Training options
param.options = trainingOptions('sgdm', ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs', 5, ...
    'Shuffle','every-epoch', ...
    'Verbose', false);
    %   'Plots', 'training-progress');
    
%% Load the dataset

digitDatasetPath = fullfile(matlabroot,'toolbox','nnet', ...
    'nndemos','nndatasets','DigitDataset');

dataset = imageDatastore(digitDatasetPath, 'IncludeSubfolders', ...
    true, 'LabelSource','foldernames');

%% Load MNIST Dataset

imgs = processImagesMNIST('train-images-idx3-ubyte');
labels = processLabelsMNIST('train-labels-idx1-ubyte');
%[imgs, labels] = readMNIST('train-images-idx3-ubyte', ...
%    'train-labels-idx1-ubyte', 60000, 0);

%dataset = read(imgs) %, labels)

%% First Train

[Nets, Subsets, Loss] = CreateNetworks(param, dataset);


% Proposed Approach

adaptive_loss = Adaptive(Nets, Subsets, Loss, param);

% Federated Approach

federated_loss = Federated(Nets, Subsets, Loss, param);

% Plot and visualize

Visualize(adaptive_loss, federated_loss, param);


