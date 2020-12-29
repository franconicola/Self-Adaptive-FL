function loss = Federated(networks, subsets, loss, param)
% FEDERATED LEARNING



% Global Aggregation
for global_it = param.globalIterations

    % Local Iterations
    for local_it = param.localIterations
            
        % Save the iteration
        iter = (global_it - 1)*param.localIterations(end) + ...
            local_it;
        
        % Initialize layer vector used for aggregation
        layers = networks{1}.Layers;

        % Industries 
        for i = 1:param.numIndustries
            
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
        
        end
        
       % Average the model    
        for l = 1:length(layers)

            % Does layer l have weights?
            if isprop(layers(l), 'Weights')

                layers(l).Weights = layers(l).Weights / param.numDevices;
            end

            % Does layer l have biases?
            if isprop(layers(l), 'Bias')

                layers(l).Bias = layers(l).Bias / param.numDevices;
            end
        end
        
        
        % Industries 
        for i = 1:param.numIndustries
            
            % Devices
            for j = 1:param.numDevices

                % Retrain Neural networks
                networks{i, j} = trainNetwork(subsets{i, j}, ...
                    layers, param.options);

                for k = 1:param.numDevices

                    % Compute the Loss for each device
                    YPred = classify(networks{i, j}, subsets{i, k});
                    YTest = subsets{i, k}.Labels;
                    loss.Function{i, (j - 1)*param.numDevices + k} = ...
                        1 - sum(YPred == YTest)/numel(YTest);
                    loss.Accuracy{i, (j - 1)*param.numDevices + k} = ...
                        sum(YPred == YTest)/numel(YTest);

                end
            end
        end
    end
end


end

