function Visualize(adaptive_loss, federated_loss, param)
% VISUALIZE 

last_step = param.localIterations(end)*param.globalIterations(end);
 
% Find the number of iterations
for iter = 1:last_step
    % Iterations of first industry and first device
    if iter == 1
        iterations{iter} = adaptive_loss.Steps{1, 1, iter};
    else
        iterations{iter} = iterations{iter - 1} + ...
            adaptive_loss.Steps{1, 1, iter};
    end
end

disp('dynamic_loss.Devices')
disp([adaptive_loss.Devices])
disp('federated_loss.Devices')
disp([federated_loss.Devices])

%% Total 

figure('Name','Loss Function for Each Industry','NumberTitle','off');

tiledlayout(2,1) % Requires R2019b or later

% Top plot
ax1 = nexttile; 
for i = 1:param.numIndustries
    plot(ax1, [iterations{:}], [adaptive_loss.AccuracyTot{i, :}], '-', ...
        'LineWidth', 2, ...
        'DisplayName', strcat('Adaptive Industry', sprintf('%2d', i)))
    hold on
    plot(ax1, [iterations{:}], [federated_loss.AccuracyTot{i, :}], '--', ...
        'LineWidth', 2, ...
        'DisplayName', strcat('Federated Industry', sprintf('%2d', i)))
end
legend('Orientation', 'horizontal', 'Location', 'southoutside', ...
    'FontSize', 10, 'NumColumns', 2)

% Bottom plot
ax2 = nexttile; 
for i = 1:param.numIndustries
    plot(ax2, [iterations{:}], [adaptive_loss.Tot{i, :}], '-', ...
        'LineWidth', 2, ... 
        'DisplayName', strcat('Adaptive Industry', sprintf('%2d', i)))
    hold on
    plot(ax2, [iterations{:}], [federated_loss.Tot{i, :}], '--', ...
        'LineWidth', 2, ... 
        'DisplayName', strcat('Federated Industry', sprintf('%2d', i)))
end

% Link the axes
linkaxes([ax1, ax2],'x'); 
xlabel('Iterations', 'FontSize', 16), xlim([0 inf])
ylabel(ax1, 'Accuracy', 'FontSize', 16), ylim(ax1, [0 1])
ylabel(ax2, 'Loss', 'FontSize', 16), ylim(ax2, [0 1]) 


disp('dynamic_loss.Tot')
disp([adaptive_loss.Tot])
disp('federated_loss.Tot')
disp([federated_loss.Tot])

fprintf('Total loss of our own method: %d \n', ...
     min([adaptive_loss.Tot{:, last_step}]));
fprintf('Total loss of federated method: %d \n', ...
     min([federated_loss.Tot{:, last_step}]));

 
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

