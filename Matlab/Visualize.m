function Visualize(dynamic_loss, federated_loss, param)
% VISUALIZE 

iterations = 1:param.localIterations(end)*param.globalIterations(end);

disp('dynamic_loss.Devices')
disp([dynamic_loss.Devices])
disp('federated_loss.Devices')
disp([federated_loss.Devices])

%% Total 

figure('Name','Loss Function for Each Industry','NumberTitle','off');

tiledlayout(2,1) % Requires R2019b or later

% Top plot
ax1 = nexttile; 
for i = 1:param.numIndustries
    plot(ax1, iterations, [dynamic_loss.AccuracyTot{i, :}], '-', ...
        'LineWidth', 2, ...
        'DisplayName', strcat('Adaptive Industry', sprintf('%2d', i)))
    hold on
    plot(ax1, iterations, [federated_loss.AccuracyTot{i, :}], '--', ...
        'LineWidth', 2, ...
        'DisplayName', strcat('Federated Industry', sprintf('%2d', i)))
end
legend('Orientation', 'horizontal', 'Location', 'southoutside', ...
    'FontSize', 10, 'NumColumns', 2)

% Bottom plot
ax2 = nexttile; 
for i = 1:param.numIndustries
    plot(ax2, iterations, [dynamic_loss.Tot{i, :}], '-', ...
        'LineWidth', 2, ... 
        'DisplayName', strcat('Adaptive Industry', sprintf('%2d', i)))
    hold on
    plot(ax2, iterations, [federated_loss.Tot{i, :}], '--', ...
        'LineWidth', 2, ... 
        'DisplayName', strcat('Federated Industry', sprintf('%2d', i)))
end

% Link the axes
linkaxes([ax1, ax2],'x'); 
xlabel('Iterations', 'FontSize', 16)
ylabel(ax1, 'Accuracy', 'FontSize', 16) 
ylabel(ax2, 'Loss', 'FontSize', 16) 

disp('dynamic_loss.Tot')
disp([dynamic_loss.Tot])
disp('federated_loss.Tot')
disp([federated_loss.Tot])

fprintf('Total loss of our own method: %d \n', ...
     min([dynamic_loss.Tot{:, iterations(end)}]));
fprintf('Total loss of federated method: %d \n', ...
     min([federated_loss.Tot{:, iterations(end)}]));

 
%% Minumum

% figure('Name','Minimum Loss of Each Industry','NumberTitle','off');
% 
% for i = 1:param.numIndustries
%     plot(iterations, [dynamic_loss.Minimum{i, :}], '-o', ...
%         'LineWidth', 2, 'MarkerSize', 10)
%     hold on
%     plot(iterations, [federated_loss.Minimum{i, :}], '-+', ...
%         'LineWidth', 2, 'MarkerSize', 10)
% end
% 
% disp('dynamic_loss.Minimum')
% disp([dynamic_loss.Minimum])
% disp('federated_loss.Minimum')
% disp([federated_loss.Minimum])
% 
% 
% fprintf('Loss of the minimum of our own method: %d \n', ...
%      min([dynamic_loss.Minimum{:, iterations(end)}]));
% fprintf('Loss of the minimum of federated method: %d \n', ...
%      min([federated_loss.Minimum{:, iterations(end)}]));

end

