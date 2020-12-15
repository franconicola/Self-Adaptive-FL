function [networks, subsets, loss] = CreateNetworks(numDevices, dataset, options)
% CREATE NETWORKS
% Creation of subsets and neural network models:
% In this example, we are using the digit dataset split into 'numDevices' 
% subsets and then we train for the first time the network onto a subset. 

    
% Split the Dataset
for i = 1:numDevices
    
    if i == 1
        [subsets{i}, subsets{i + 1}] = splitEachLabel(dataset, 1 / numDevices);
    elseif i < numDevices
        [subsets{i}, subsets{i + 1}] = splitEachLabel(subsets{i}, ...
            1 / (numDevices - i + 1));
    end
end


% Create Neural Networks
for i = 1:numDevices
    
    % Layers definition
    layers{i} = [ imageInputLayer([28 28 1])
    convolution2dLayer(5,20)
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];
    
    % Train the network
    networks{i} = trainNetwork(subsets{i}, layers{i}, options);
    
    % Provide the first Loss
    for j = 1:numDevices
        YPred = classify(networks{i}, subsets{j});
        YTest = subsets{j}.Labels;
        loss{(i - 1)*numDevices + j} = 1-sum(YPred == YTest)/numel(YTest);
    end
    
end


end

