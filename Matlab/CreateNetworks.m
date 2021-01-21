function [networks, subsets, loss] = CreateNetworks(param, dataset)

% CREATE NETWORKS
% Creation of subsets and neural network models:
% In this example, we are using the digit dataset split into 'numDevices' 
% subsets and then we train for the first time the network on a subset. 


% Split dataset between Factories
      
for i = 1:param.numFactories

    if i == 1
        [factory{1}, factory{2}] = ...
            splitEachLabel(dataset, 1 / param.numFactories, 'randomize');
    elseif i < param.numFactories
        [factory{i}, factory{i + 1}] = ...
            splitEachLabel(factory{i}, ...
            1 / (param.numFactories - i + 1), 'randomize');
    end

end

% Split dataset between Devices
% subsets{i, j} is the subset of Factories i and device j
for i = 1:param.numFactories
    for j = 1:param.numDevices
    
        if j == 1
            [subsets{i, 1}, subsets{i, 2}] = ...
                splitEachLabel(factory{i}, 1 / param.numDevices, ...
                'randomize');
        elseif j < param.numDevices
            [subsets{i, j}, subsets{i, j + 1}] = ...
                splitEachLabel(subsets{i, j}, ...
                1 / (param.numDevices - j + 1), 'randomize');
        end

    end
end


% Layers definition
layers = [
    imageInputLayer([28 28 1])

    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer

    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];
        
pre_options = trainingOptions('sgdm', ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs', 2, ...
    'Shuffle','every-epoch', ...
    'Verbose', false);

[network, options] = trainNetwork(dataset, layers, pre_options);

% Create Neural Networks
for i = 1:param.numFactories
    for j = 1:param.numDevices
            
        
        % Train the network
        networks{i, j} = network;
        
        % Store the number of steps
        loss.Steps{i, j, 1} = size(options.TrainingLoss, 2);


        % Provide the first Loss
        for k = 1:param.numDevices
            YPred = classify(networks{i, j}, subsets{i, k});
            YTest = subsets{i, k}.Labels;
            loss.Function{i, (j - 1)*param.numDevices + k} = ...
                1-sum(YPred == YTest)/numel(YTest);
            loss.Accuracy{i, (j - 1)*param.numDevices + k} = ...
                sum(YPred == YTest)/numel(YTest);
        end
    end
end


% % Create Neural Networks
% for i = 1:param.numFactories
%     for j = 1:param.numDevices
% 
% 
%         % Layers definition
%         layers = [
%             imageInputLayer([28 28 1])
% 
%             convolution2dLayer(3,8,'Padding','same')
%             batchNormalizationLayer
%             reluLayer
% 
%             maxPooling2dLayer(2,'Stride',2)
% 
%             convolution2dLayer(3,16,'Padding','same')
%             batchNormalizationLayer
%             reluLayer
% 
%             maxPooling2dLayer(2,'Stride',2)
% 
%             convolution2dLayer(3,32,'Padding','same')
%             batchNormalizationLayer
%             reluLayer
% 
%             fullyConnectedLayer(10)
%             softmaxLayer
%             classificationLayer];
%         
%         % Train the network
%         [networks{i, j}, options] = ... 
%             trainNetwork(subsets{i, j}, layers, param.options);
%         
%         % Store the number of steps
%         loss.Steps{i, j, 1} = size(options.TrainingLoss, 2);
% 
% 
%         % Provide the first Loss
%         for k = 1:param.numDevices
%             YPred = classify(networks{i, j}, subsets{i, k});
%             YTest = subsets{i, k}.Labels;
%             loss.Function{i, (j - 1)*param.numDevices + k} = ...
%                 1-sum(YPred == YTest)/numel(YTest);
%             loss.Accuracy{i, (j - 1)*param.numDevices + k} = ...
%                 sum(YPred == YTest)/numel(YTest);
%         end
%     end
% end

end

