clc;
clear;
close all;

%% Parameters

% Number of devices
numDevices = 4;

% Order of subsets
subsetsOrder = 1:numDevices;


%% First Train

[Nets, Subsets, Loss] = CreateNetworks(numDevices);

fprintf('Accuracy at first step: %d\n', sum([Loss{:}]));


%% Switch the dataset and train

for i = 1:numDevices - 1
    

    % Discriminant own loss
    for j = 1:numDevices
        disp([Loss{1 + (j - 1)*numDevices: ...
        numDevices + (j - 1)*numDevices}])
        fprintf('Accuracy at %d step of agent %d: %d\n', i, j, ...
        sum([Loss{1 + (j - 1)*numDevices: ...
        numDevices + (j - 1)*numDevices}]));
    
        Loss{subsetsOrder(j) + (j - 1)*numDevices} = 1;
    end
        
    % Linear Programming
    subsetsOrder = LinearProgramming(numDevices, Loss)
    
    % Reassign subsets 
    temporarySubsets = Subsets;

    for j = 1:numDevices
        temporarySubsets{j} = Subsets{subsetsOrder(j)};
    end

    % Retrain the Networks 
    [Nets, Loss] = TrainNetworks(Nets, temporarySubsets, numDevices);



end

