function loss = Federated(networks, subsets, loss, param)
% FEDERATED LEARNING

% Global Aggregation
for global_it = param.globalIterations

    % Local Iterations
    for local_it = param.localIterations
            
        % Save the iteration
        iter = (global_it - 1)*param.localIterations(end) + ...
            local_it;

        % Industries 
        for i = 1:param.numIndustries
            
            % Initialize layer vector used for aggregation
            layers = networks{i, 1}.Layers;
            
            % Devices
            for j = 1:param.numDevices
            
                % Device loss
                loss.Devices{i, j, iter} = ...
                    sum([loss.Function{i, 1 + (j - 1)*param.numDevices: ... 
                    param.numDevices + (j - 1)*param.numDevices}]) ...
                    / param.numDevices;
            
                % Minimum loss
                if j == 1
                    
                    loss.Minimum{i, iter} = ...
                        loss.Devices{i, 1, iter};
                    
                elseif loss.Devices{i, j, iter} < loss.Minimum{i, iter}
                    
                    loss.Minimum{i, iter} = loss.Devices{i, j, iter};
                end
                
                % Federated Learning sum the weights
                if j > 1
                    for l = 1:length(layers)

                        % Does layer l have weights?
                        if isprop(layers(l), 'Weights') 
                            layers(l).Weights = layers(l).Weights + ...
                                networks{i, j}.Layers(l).Weights;
                        end

                        % Does layer l have biases?
                        if isprop(layers(l), 'Bias')
                            layers(l).Bias = layers(l).Bias + ...
                                networks{i, j}.Layers(l).Bias;
                        end
                    end
               
                end
                
            end
            
            % Total loss
            loss.Tot{i, iter} = ...
                sum([loss.Devices{i, :, iter}]) / param.numDevices;
            % Total Accuracy
            loss.AccuracyTot{i, iter} = ...
                sum([loss.Accuracy{i, :}]) / param.numDevices^2;
            
                
            % Average the model    
            for l = 1:length(layers)

                % Does layer l have weights?
                if isprop(layers(l), 'Weights')

                    layers(l).Weights = ...
                        layers(l).Weights / param.numDevices;
                end

                % Does layer l have biases?
                if isprop(layers(l), 'Bias')

                    layers(l).Bias = ...
                        layers(l).Bias / param.numDevices;
                end
            end
            
            % Devices
            for j = 1:param.numDevices

                % Retrain Neural networks
                [networks{i, j}, options] = ... 
                    trainNetwork(subsets{i, j}, layers, ...
                    param.options);


                % Store the number of steps
                loss.Steps{i, j, iter} = ...
                    size(options.TrainingLoss, 2);

                % Iterate through the dataset 
                for k = 1:param.numDevices

                    % Test the trained network of device j 
                    % through the entire data set
                    YPred = classify(networks{i, j}, ...
                        subsets{i, k});
                    
                    YTest = subsets{i, k}.Labels;
                    
                    % Compute the loss 
                    loss.Function{i, (j - 1)*...
                        param.numDevices + k} = ...
                        1 - sum(YPred == YTest)/numel(YTest);
                    
                    % Compute the accuracy
                    loss.Accuracy{i, (j - 1)*...
                        param.numDevices + k} = ...
                        sum(YPred == YTest)/numel(YTest);

                end
            end
            
        end
        
    end
    
end


end

