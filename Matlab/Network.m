clc;
clear;
close all;

%% Parameters

% Number of devices
numDev = 5;

% Order of subsets
subsetsOrder = 1:numDev;

% Training options
options = trainingOptions('sgdm', 'MaxEpochs', 30,...
    'InitialLearnRate',1e-4, 'Verbose',false);
    %'Plots', 'training-progress');
    
% Load the dataset
digitDatasetPath = fullfile(matlabroot,'toolbox','nnet', ...
    'nndemos','nndatasets','DigitDataset');
dataset = imageDatastore(digitDatasetPath, 'IncludeSubfolders', ...
    true, 'LabelSource','foldernames');


%% First Train

[Nets, Subsets, Loss] = CreateNetworks(numDev, dataset, options);


%% Proposed Approach

% Initialization of the networks after the first training
p_Nets = Nets;
p_Loss = Loss;

% Loss function for the devices
p_loss_dev = zeros(numDev, numDev);

for i = 1:numDev
    
    for j = 1:numDev
        
        % Compute devices loss
        p_loss_dev(i, j) = ...
            sum([p_Loss{1 + (j - 1)*numDev:numDev + (j - 1)*numDev}]);
        
        % Discriminant own loss    
        p_Loss{subsetsOrder(j) + (j - 1)*numDev} = 1;
    end
    
    if i < numDev
        % Linear Programming
        subsetsOrder = LinearProgramming(numDev, p_Loss);

        % Initialize temporary subsets 
        tempSubsets = Subsets;
        
         
        for j = 1:numDev
            % Reassign subsets
            tempSubsets{j} = Subsets{subsetsOrder(j)};

            % Retrain Neural networks            
            p_Nets{j} = trainNetwork(tempSubsets{j}, ...
                p_Nets{j}.Layers, options);
            

            YPred = classify(p_Nets{j}, );
            YTest = Subsets{k}.Labels;
            p_Loss{(j - 1)*numDev + k} = 1-sum(YPred == YTest)/numel(YTest);

        end 
        
    end
    
end


%% Federated Approach

% Initialization of the networks after the first training
f_Nets = Nets;
f_Subsets = Subsets;
f_Loss = Loss;

% Loss function for the devices
f_loss_dev = zeros(numDev, numDev);

% Ordered number of subsets
subsetsOrder = 1:numDev;

for i = 1:numDev
    
    % Initialize layer vector used for aggregation
    layers = Nets{1}.Layers;
    
    for j = 1:numDev
        
        % Computing device loss        
        f_loss_dev(i, j) = ...
            sum([f_Loss{1 + (j - 1)*numDev:numDev + (j - 1)*numDev}]);
        
        % Switch to the next subset
        if subsetsOrder(j) + 1 > numDev 
            subsetsOrder(j) = 1;
        else
            subsetsOrder(j) = subsetsOrder(j) + 1;
        end
        
        % Federated Learning sum the weights
        if j > 1
            for l = 1:length(layers)
                if isprop(layers(l), 'Weights') % Does layer l have weights?
                    layers(l).Weights = layers(l).Weights + ...
                        Nets{j}.Layers(l).Weights;
                end
                if isprop(layers(l), 'Bias')%  Does layer l have biases?
                    layers(l).Bias = layers(l).Bias + Nets{j}.Layers(l).Bias;
                end
            end
        end
    end
    
    % Federated Learning build the model into the first device    
    for l = 1:length(layers)
        if isprop(layers(l), 'Weights') % Does layer l have weights?
            layers(l).Weights = layers(l).Weights / numDev;
        end
        if isprop(layers(l), 'Bias')% Does layer l have biases?
            layers(l).Bias = layers(l).Bias / numDev;
        end
    end
    
    if i < numDev

        % Initialize temporary subsets 
        tempSubsets = Subsets;

        for j = 1:numDev
            tempSubsets{j} = Subsets{subsetsOrder(j)};
        end

        % Retrain Neural networks
        for j = 1:numDev

            f_Nets{j} = trainNetwork(tempSubsets{j}, layers, options);

            for k = 1:numDev
                    YPred = classify(f_Nets{j}, Subsets{k});
                    YTest = Subsets{k}.Labels;
                    f_Loss{(j - 1)*numDev + k} = ...
                        1 - sum(YPred == YTest)/numel(YTest);
            end
        end
    end
end

%% Plot and visualize

iterations = 1:numDev;
figure 
for i = 1:numDev
    plot(iterations(i), p_loss_dev(i,:), 'o')
    hold on
    plot(iterations(i), f_loss_dev(i,:), '+')
end


disp(p_loss_dev)
disp(f_loss_dev)

fprintf('Loss of the minimum of our own method: %d \n', ...
     min(p_loss_dev(numDev, :)));
fprintf('Loss of the minimum of federated method: %d \n', ...
     min(f_loss_dev(numDev, :)));



