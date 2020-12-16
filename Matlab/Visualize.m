function Visualize(dynamic_loss, federated_loss, param)
% VISUALIZE 

iterations = 1:param.localIterations(end) + param.globalIterations(end);

figure 

for i = 1:param.numIndustries
    for j = 1:param.numDevices
        
        iter = (i - 1)*param.numDevices + j;

        plot(iter, [dynamic_loss.Devices{i, :, iter}], 'o')
        hold on
        plot(iter, [federated_loss.Devices{i, :, iter}], '+')
        hold on
    end
end

disp([dynamic_loss.Devices])
disp([federated_loss.Devices])


figure 
for i = iterations
    plot(iterations(i), [dynamic_loss.Tot{:, i}], 'o')
    hold on
    plot(iterations(i), [federated_loss.Tot{:, i}], '+')
    hold on
end

disp([dynamic_loss.Tot])
disp([federated_loss.Tot])

fprintf('Total loss of our own method: %d \n', ...
     min(dynamic_loss.Tot{:, iterations(end)}));
fprintf('Total loss of federated method: %d \n', ...
     min(federated_loss.Tot{:, iterations(end)}));


figure 
for i = iterations
    plot(iterations(i), [dynamic_loss.Minimum{:, i}], 'o')
    hold on
    plot(iterations(i), [federated_loss.Minimum{:, i}], '+')
end

disp([dynamic_loss.Minimum])
disp([federated_loss.Minimum])


fprintf('Loss of the minimum of our own method: %d \n', ...
     min(dynamic_loss.Minimum{:, iterations(end)}));
fprintf('Loss of the minimum of federated method: %d \n', ...
     min(federated_loss.Minimum{:, iterations(end)}));

end

