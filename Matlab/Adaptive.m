function loss = Adaptive(networks, subsets, loss, param)

% SELF-ADAPTIVE
% Self-adaptive approach is the proposed approach


% Global Aggregation
for global_it = param.globalIterations
    
    % Store the number of global iterations
    g_step = global_it*param.localIterations(end);

    % Factories
    for i = 1:param.numFactories
        
        % Order of subsets
        subsetsOrder = 1:param.numDevices;
        
        % Local Iterations
        for local_it = param.localIterations
            
            % Store the number of local iterations
            iter = (global_it - 1)*param.localIterations(end) + ...
                local_it;
            
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
                    
                    % Store the best one
                    loss.MinDevice{i} = 1;
                    
                elseif loss.Devices{i, j, iter} < loss.Minimum{i, iter}

                    loss.Minimum{i, iter} = loss.Devices{i, j, iter};
                    
                    % Store the best one
                    loss.MinDevice{i} = j;

                end
               
                
                % Discriminant own loss    
                loss.Function{i, ...
                    subsetsOrder(j) + (j - 1)*param.numDevices} = 1;
            end
                
            
            % Total loss        
            loss.Tot{i, iter} = ...
                sum([loss.Devices{i, :, iter}]) / param.numDevices;
            loss.AccuracyTot{i, iter} = ...
                sum([loss.Accuracy{i, :}]) / param.numDevices^2;

            % Not train in the last step
            if local_it < param.localIterations(end)

                % Linear Programming
                subsetsOrder = LinearProgramming(param.numDevices, ...
                    [loss.Function{i, :}]);

                
                % Initialize temporary subsets 
                tempSubsets = subsets;


                for j = 1:param.numDevices
                    
                    % Reassign subsets
                    tempSubsets{i, j} = subsets{i, subsetsOrder(j)};

                    % Retrain Neural networks
                    [networks{i, j}, options] = ...
                        trainNetwork(tempSubsets{i, j}, ...
                        networks{i, j}.Layers, param.options);
                    
                    % Store the number of steps
                    loss.Steps{i, j, iter} = size(options.TrainingLoss, 2);


                    % Individual Subsets Loss funtions
                    for k = 1:param.numDevices
                        YPred = classify(networks{i, j}, subsets{i, k});
                        YTest = subsets{i, k}.Labels;
                        loss.Function{i, ...
                            (j - 1)*param.numDevices + k} = ...
                            1 - sum(YPred == YTest) / numel(YTest);
                        loss.Accuracy{i, ...
                            (j - 1)*param.numDevices + k} = ...
                            sum(YPred == YTest) / numel(YTest);

                    end

                end 
                
            end
            
        end
        
    end
    
    
    
    
    
    
    
    
    % Federated Learning

    % Initialize layer vector used for aggregation
    layers = networks{1, loss.MinDevice{1}}.Layers;
    
    % Factories 
    for i = 1:param.numFactories

        for l = 1:length(layers)

            % Does layer l have weights?
            if isprop(layers(l), 'Weights') 
                layers(l).Weights = layers(l).Weights + ...
                    networks{i, loss.MinDevice{i}}.Layers(l).Weights;
            end

            % Does layer l have biases?
            if isprop(layers(l), 'Bias')
                layers(l).Bias = layers(l).Bias + ...
                    networks{i, loss.MinDevice{i}}.Layers(l).Bias;

            end

        end
        
    end
    
   % Average the model    
    for l = 1:length(layers)
        
        % Does layer l have weights?
        if isprop(layers(l), 'Weights')
            
            layers(l).Weights = layers(l).Weights / param.numFactories;
        end
        
        % Does layer l have biases?
        if isprop(layers(l), 'Bias')
        
            layers(l).Bias = layers(l).Bias / param.numFactories;
        end
    end

    % Factories 
    for i = 1:param.numFactories

        % Devices
        for j = 1:param.numDevices
        
            % Retrain Neural networks
            [networks{i, j}, options] = ... 
                trainNetwork(subsets{i, j}, layers, param.options);
            
            % Store the number of steps
            loss.Steps{i, j, g_step} = size(options.TrainingLoss, 2);
            
            for k = 1:param.numDevices

                % Test the trained network of device j 
                % through the entire data set
                YPred = classify(networks{i, j}, subsets{i, k});
                % Correspondingly Labels
                YTest = subsets{i, k}.Labels;
                
                % Compute the loss 
                loss.Function{i, (j - 1)*param.numDevices + k} = ...
                    1 - sum(YPred == YTest)/numel(YTest);
                
                % Compute the accuracy
                loss.Accuracy{i, (j - 1)*param.numDevices + k} = ...
                    sum(YPred == YTest)/numel(YTest);

            end
            
        end
       
    end
    
end
    
    

end

