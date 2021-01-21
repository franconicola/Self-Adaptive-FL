function Visualize(adaptive, federated, param)
% VISUALIZE 

last_step = param.localIterations(end)*param.globalIterations(end);
 
% Find the number of iterations
adap_steps = [adaptive.Steps{1, 1, :}];
federated_steps = [federated.Steps{1, 1, :}];

for iter = 2:last_step
    
    % Iterations of first factory and first device
    adap_steps(iter) = adap_steps(iter) + adap_steps(iter - 1);
    federated_steps(iter) = federated_steps(iter) + federated_steps(iter - 1);
    
end

% Initialize the mean of the loss and accuracy
adaptive_acc_mean = zeros(1, last_step);
adaptive_loss_mean = zeros(1, last_step);
federated_acc_mean = zeros(1, last_step);
federated_loss_mean = zeros(1, last_step);

% Compute the mean of the loss and accuracy
for iter = 1:last_step
    
    adaptive_acc_mean(iter) = sum([adaptive.AccuracyTot{:, iter}]) / ...
        param.numFactories;

    adaptive_loss_mean(iter) = sum([adaptive.Tot{:, iter}]) / ...
        param.numFactories;
    
    federated_acc_mean(iter) = sum([federated.AccuracyTot{:, iter}]) / ...
        param.numFactories;
    
    federated_loss_mean(iter) = sum([federated.Tot{:, iter}]) / ...
        param.numFactories;

end

%% Mean Plot 

figure('Name', 'Mean', 'NumberTitle', 'off');

tiledlayout(2,1) % Requires R2019b or later

% Top plot 
ax1 = nexttile; 

% Adaptive accuracy plot
plot(ax1, adap_steps, adaptive_acc_mean, '-ko', 'LineWidth', 2, ...
    'DisplayName', 'Self-Adaptive')

hold on

% Federated accuracy plot
plot(ax1, federated_steps, federated_acc_mean, '-.bx', ...
    'LineWidth', 2, 'DisplayName', 'Federated Averaging')

% Legend
legend('Orientation', 'horizontal', 'Location', 'northoutside', ...
    'FontSize', 11, 'NumColumns', 3)
 
% Bottom plot
ax2 = nexttile; 

% Adaptive loss plot
plot(ax2, adap_steps, adaptive_loss_mean, '-ko', 'LineWidth', 2, ...
    'DisplayName', 'Self-Adaptive')

hold on

% Federated loss plot
plot(ax2, federated_steps, federated_loss_mean, '-.bx', ...
    'LineWidth', 2, 'DisplayName', 'Federated Averaging')



% Link the axes
linkaxes([ax1, ax2],'x'); 
xlabel('Iterations', 'FontSize', 16), xlim([0 inf])
ylabel(ax1, 'Accuracy', 'FontSize', 16), ylim(ax1, [0.9 1])
ylabel(ax2, 'Loss', 'FontSize', 16), ylim(ax2, [0 0.1]) 


% Print Results
fprintf('Accuracy of our own method: %d \n', ...
    adaptive_acc_mean(round(last_step / 2)));
fprintf('Accuracy of federated averaging method: %d \n', ...
    federated_acc_mean(round(last_step / 2)));


% %% Second Plot
%  
% figure('Name','Each factory','NumberTitle','off');
% 
% tiledlayout(2,1) % Requires R2019b or later
% 
% 
% % Top plot 
% ax1_2 = nexttile; 
% 
% for i = 1:param.numFactories
%     % Adaptive accuracy plot
%     plot(ax1_2, adap_steps, [adaptive.AccuracyTot{i, :}], ...
%         '-', 'LineWidth', 2, 'DisplayName', ...
%         strcat('Self-Adaptive Factory', sprintf('%2d', i)))
%     hold on
% end
% 
% % Federated accuracy plot
% plot(ax1_2, federated_steps, federated_acc_mean, '-', ...
%     'LineWidth', 2, 'DisplayName', 'Baseline 1')
% 
%     
% legend('Orientation', 'horizontal', 'Location', 'northoutside', ...
%     'FontSize', 12, 'NumColumns', 2)
%  
% % Bottom plot
% ax2_2 = nexttile; 
% 
% % Adaptive loss plot
% for i = 1:param.numFactories
%     plot(ax2_2, adap_steps, [adaptive.Tot{i, :}], '-', ...
%         'LineWidth', 2, 'DisplayName', ...
%         strcat('Self-Adaptive Smart Factory', sprintf('%2d', i)))
%     hold on
% end
% 
% 
% % Federated loss plot
% plot(ax2_2, federated_steps, federated_loss_mean, '-', ...
%         'LineWidth', 2, 'DisplayName', 'Baseline 1')
% 
% 
% % Link the axes
% linkaxes([ax1_2, ax2_2],'x'); 
% xlabel('Iterations', 'FontSize', 16), xlim([0 inf])
% ylabel(ax1_2, 'Accuracy', 'FontSize', 16), ylim(ax1, [0.9 1])
% ylabel(ax2_2, 'Loss', 'FontSize', 16), ylim(ax2, [0 0.1]) 


end

