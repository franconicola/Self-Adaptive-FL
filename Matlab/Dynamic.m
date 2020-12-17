function loss = Dynamic(networks, subsets, loss, param)
% DYNAMIC 
% Dynamic approach is the proposed approach
% Detailed explanation goes here



% Global Aggregation
for global_it = param.globalIterations
    %% Dynamic Framework

    % Industries 
    for i = 1:param.numIndustries
        
        % Order of subsets
        subsetsOrder = 1:param.numDevices;
        
        % Local Iterations
        for local_it = param.localIterations
            
            % Save the iteration
            
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

            % TODO: Give me a definition of this condition...
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
                    networks{i, j} = trainNetwork(tempSubsets{i, j}, ...
                        networks{i, j}.Layers, param.options);

                    % Individual Subsets Loss funtions
                    for k = 1:param.numDevices
                        YPred = classify(networks{i, j}, subsets{i, k});
                        YTest = subsets{i, k}.Labels;
                        loss.Function{i, ...
                            (j - 1)*param.numDevices + k} = ...
                            1 - sum(YPred == YTest) / numel(YTest);

                    end

                end 
                
            end
            
        end
        
    end
    
    
    
    
    
    
    
    
    %% % Federated Learning
    
    % TODO: Should provide the minimum model???

    % Industries 
    for i = 1:param.numIndustries

    
        % Federated Learning sum the weights
        if i == 1      
            
            % Initialize layer vector used for aggregation
            layers = networks{1, loss.MinDevice{1}}.Layers;
            
        else
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

            end
            
        end
       
    end
    
end
    
    

end

