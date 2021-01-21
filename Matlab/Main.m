% Requires R2019b or later


close all; 
clear, clc

%% Parameters

% Number of factories
param.numFactories = 4;

% Number of devices
param.numDevices = 20;

% Number of local iterations
param.localIterations = 1:10;

% Number of global iterations
param.globalIterations = 1:10;

% Training options
param.options = trainingOptions('sgdm', ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs', 30, ...
    'Shuffle','every-epoch', ...
    'Verbose', false);
    
%% Load the dataset

digitDatasetPath = fullfile(matlabroot,'toolbox','nnet', ...
    'nndemos','nndatasets','DigitDataset');

dataset = imageDatastore(digitDatasetPath, 'IncludeSubfolders', ...
    true, 'LabelSource','foldernames');


%% First Train
disp('Create the network and split the dataset...')
[nets, subsets, loss] = CreateNetworks(param, dataset);
disp('OK')

% Proposed Approach
disp('Compute the adaptive approach...')
adaptive = Adaptive(nets, subsets, loss, param);
disp('OK')

% Federated Approach
disp('Compute the first baseline approach...')
federated = Federated(nets, subsets, loss, param);
disp('OK')


% Plot and visualize
disp('Plot the results')
Visualize(adaptive, federated, param);


