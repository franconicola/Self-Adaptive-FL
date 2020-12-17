clc;
clear;
close all;

%% Parameters

% Number of industries
param.numIndustries = 2;

% Number of devices
param.numDevices = 5;

% Number of local iterations
param.localIterations = 1:param.numDevices;

% Number of global iterations
param.globalIterations = 1:param.numIndustries;


% Training options
param.options = trainingOptions('sgdm', 'MaxEpochs', 5,...
    'InitialLearnRate',1e-4, 'Verbose',false);
    %'Plots', 'training-progress');
    
% Load the dataset
digitDatasetPath = fullfile(matlabroot,'toolbox','nnet', ...
    'nndemos','nndatasets','DigitDataset');

dataset = imageDatastore(digitDatasetPath, 'IncludeSubfolders', ...
    true, 'LabelSource','foldernames');


%% First Train

[Nets, Subsets, Loss] = CreateNetworks(param, dataset);


%% Proposed Approach

dynamic_loss = Dynamic(Nets, Subsets, Loss, param);

%% Federated Approach

federated_loss = Federated(Nets, Subsets, Loss, param);

%% Plot and visualize

Visualize(dynamic_loss, federated_loss, param);


