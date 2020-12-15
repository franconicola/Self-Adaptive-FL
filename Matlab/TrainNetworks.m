function [networks, loss] = TrainNetworks(networks, subsets, numDevices)
% TRAIN networksWORKS 
% Train the neural networksworks
% This function train neural networks

% Training options
options = trainingOptions('sgdm', 'MaxEpochs', 5,...
    'InitialLearnRate',1e-4, 'Verbose',false);
    %'Plots', 'training-progress');


% Retrain Neural networks
for i = 1:numDevices
    
    if i == numDevices
        networks{i} = trainNetwork(subsets{1}, ...
            networks{i}.Layers, options);
    else
        networks{i} = trainNetwork(subsets{i + 1}, ...
            networks{i}.Layers, options);
    end
    
    for j = 1:numDevices
        YPred = classify(networks{i}, subsets{j});
        YTest = subsets{j}.Labels;
        loss{(i - 1)*numDevices + j} = 1-sum(YPred == YTest)/numel(YTest);
    end

end 


end

