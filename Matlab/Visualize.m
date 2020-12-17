function Visualize(dynamic_loss, federated_loss, param)
% VISUALIZE 

iterations = 1:param.localIterations(end)*param.globalIterations(end);

figure('Name', 'Loss Fuction For Each Device', 'NumberTitle','off');

for i = 1:param.numIndustries
    for j = 1:param.numDevices
        
        iter = (i - 1)*param.numDevices + j;

        plot(iter, [dynamic_loss.Devices{i, :, iter}], 'o',  ...
            'DisplayName', 'Proposed Loss Devices')
        
        hold on
        plot(iter, [federated_loss.Devices{i, :, iter}], 'x',  ...
            'DisplayName', 'Federated Loss Devices')
    end
end

disp('dynamic_loss.Devices')
disp([dynamic_loss.Devices])
disp('federated_loss.Devices')
disp([federated_loss.Devices])


figure('Name','Loss Function for Each Industry','NumberTitle','off');

for i = 1:param.numIndustries
    plot(iterations, [dynamic_loss.Tot{i, :}], '-o')
    hold on
    plot(iterations, [federated_loss.Tot{i, :}], '-+')
end

disp('dynamic_loss.Tot')
disp([dynamic_loss.Tot])
disp('federated_loss.Tot')
disp([federated_loss.Tot])

fprintf('Total loss of our own method: %d \n', ...
     min([dynamic_loss.Tot{:, iterations(end)}]));
fprintf('Total loss of federated method: %d \n', ...
     min([federated_loss.Tot{:, iterations(end)}]));


figure('Name','Minimum Loss of Each Industry','NumberTitle','off');

for i = 1:param.numIndustries
    plot(iterations, [dynamic_loss.Minimum{i, :}], '-o')
    hold on
    plot(iterations, [federated_loss.Minimum{i, :}], '-+')
end

disp('dynamic_loss.Minimum')
disp([dynamic_loss.Minimum])
disp('federated_loss.Minimum')
disp([federated_loss.Minimum])


fprintf('Loss of the minimum of our own method: %d \n', ...
     min([dynamic_loss.Minimum{:, iterations(end)}]));
fprintf('Loss of the minimum of federated method: %d \n', ...
     min([federated_loss.Minimum{:, iterations(end)}]));

end

