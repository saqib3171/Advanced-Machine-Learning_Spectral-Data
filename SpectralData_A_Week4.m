clc;
clear all;
close all;
warning off;

%% Week 2

% Define the filenames for data files
filename = 'data_part_1.csv';
filename1 = 'data_part_2.xlsx';

% Read data from CSV and Excel files into variables
data = csvread(filename, 1, 1); % Skip the header row and first column
data1 = xlsread(filename1);

% Select traits from dataset1 for the boxplot
selected_variables = data(:, 2:21);
variable_names = {'Anthocyanin content','Boron content', 'C content', 'Ca content', 'Carotenoid content', 'Cellulose', 'Chl content ','Copper content', 'EWT', 'Fiber', 'LAI', 'LMA', 'Lignin','Magnesium content', 'Manganese content', 'N content', 'NSC', 'Phosphorus content', 'Potassium content', 'Sulfur content'};

% Create a figure for the boxplot
figure;
boxplot(selected_variables, 'Colors', 'bgrm','Notch', 'on', 'OutlierSize', 5,'Labels', variable_names);
xlabel('Variables (Predictors)');
ylabel('Values');
title('Boxplot of Variables from dataset1');
xtickangle(45);
set(gcf, 'Position', [100, 100, 800, 600]);

% Select traits from dataset2 for the boxplot
selected_variables2 = data1(:, 2:38);
variable_names2 = {'Anthocyanin content','Anthocyanin concentration', 'Boron content','Boron concentration ', 'C content', 'C concentration', 'Ca content','Ca concentration','Carotenoid content', 'Carotenoid concentration', 'Cellulose (mg/cmÂ²)', 'Cellulose (mg/g)','Chl content (Î¼g/cmÂ²)','Chlorophyll concentration (mg/g)', 'Copper content (mg/cmÂ²)','Copper concentration (mg/g)', 'EWT (mg/cmÂ²)', 'Fiber (mg/cmÂ²)', 'Fiber (mg/g)','LAI (mÂ²/mÂ²)', 'LMA (g/mÂ²)', 'Lignin (mg/cmÂ²)', 'Lignin (mg/g)', 'Magnesium content (mg/cmÂ²)', 'Magnesium concentration (mg/g)', 'Manganese content (mg/cmÂ²)','Manganese concentration (mg/g)', 'N content (mg/cmÂ²)', 'N concentration (mg/g)', 'NSC (mg/cmÂ²)','NSC (mg/g)', 'Phosphorus content (mg/cmÂ²)', 'P concentration (mg/g)', 'Potassium content (mg/cmÂ²)','Potassium concentration (mg/g)', 'Sulfur content (mg/cmÂ²)','Sulfur concentration (mg/g)'};

% Create a figure for the boxplot
figure;
boxplot(selected_variables2, 'Colors', 'bgrm','Notch', 'on', 'OutlierSize', 5,'Labels', variable_names2);
xlabel('Variables (Predictors)');
ylabel('Values');
title('Boxplot of Variables from dataset2');
xtickangle(45);
set(gcf, 'Position', [100, 100, 800, 600]);

% Check for missing values in dataset1 and dataset2
missing_values = isnan(data);
missing_values1 = isnan(data1);

fprintf('Missing values in dataset1:\n');
disp(missing_values);
fprintf('Missing values in dataset2:\n');
disp(missing_values1);

% Display and save the surface plots with labels
figure;
subplot(1,2,1);
surf(data(2:50,22:end));
xlabel('X-axis');
ylabel('Y-axis');
zlabel('Z-axis');
title('Surface Plot of dataset1');
subplot(1,2,2);
imagesc(data(2:50,22:end));
xlabel('X-axis');
ylabel('Y-axis');
title('Image Plot of dataset1');
colormap hot;
set(gcf, 'Position', [100, 100, 800, 600]);
%saveas(gca, f1.png');

figure;
imagesc(data1(2:50,31:end));
xlabel('X-axis');
ylabel('Y-axis');
title('Image Plot of dataset2');
colormap hot;
set(gcf, 'Position', [100, 100, 800, 600]);
%saveas(gca,f2.png');

% Initialize an empty array for FFT results
fftarr = [];

% Display FFT plots
figure;
hold on;
for i = 100:120
    p = abs(fftshift(fft(data(i,22:end))));
    plot(p);
    pause(0.01);
    fftarr = [fftarr; p];
end

fprintf('FFT plots for rows 100 to 120 completed.\n');

figure; % carbon
plot(data(2:end,4));
xlabel('X-axis');
ylabel('Y-axis');
title('Carbon Plot');

fprintf('Carbon plot completed.\n');

figure; % cellulose
plot(data(2:end,6));
xlabel('X-axis');
ylabel('Y-axis');
title('Cellulose Plot');

fprintf('Cellulose plot completed.\n');

%% Week 3

% Multivariate Regression
X = data(:, 2:21); % Independent variables (known traits)
Y = data(:, 22:end); % Dependent variables (traits to predict)

% Initialize matrices for predictions and prediction intervals
predicted_traits = NaN(size(Y));
prediction_intervals = NaN(size(Y, 1), size(Y, 2), 2);

% Create a scatter plot matrix
figure;
numVars = size(X, 2);

for i = 1:numVars
    for j = 1:numVars
        subplot(numVars, numVars, (i - 1) * numVars + j);
        if i == j
            % Plot histograms on the diagonal
            histogram(X(:, i));
        else
            % Create scatter plots for off-diagonal elements
            scatter(X(:, j), X(:, i), 'filled');
        end
        
        % Add axis labels only to the outer subplots
        if i == numVars
            xlabel(['X' num2str(j)]);
        end
        if j == 1
            ylabel(['X' num2str(i)]);
        end
    end
end

% Add a title to the entire plot 
sgtitle('Scatter Plot Matrix of Independent Variables');

% Create a scatter plot matrix
figure;
numVars = size(X, 2);

for i = 1:numVars
    for j = 1:numVars
        subplot(numVars, numVars, (i - 1) * numVars + j);
        if i == j
            % Plot histograms on the diagonal
            histogram(X(:, i));
        else
            % Create scatter plots for off-diagonal elements
            scatter(X(:, j), X(:, i), 'filled');
        end
        
        % Add axis labels only to the outer subplots
        if i == numVars
            xlabel(['X' num2str(j)]);
        end
        if j == 1
            ylabel(['X' num2str(i)]);
        end
    end
end


% Loop through each trait to predict
for i = 1:size(Y, 2)
    current_trait = Y(:, i);
    missing_indices = isnan(current_trait);
    X_known = X(~missing_indices, :);
    Y_known = current_trait(~missing_indices);
    
    % Perform multivariate regression
    mdl = fitlm(X_known, Y_known);
    
    % Predict the current trait for all data points
    predicted_values = predict(mdl, X);
    predicted_traits(:, i) = predicted_values;
    
    % Calculate prediction intervals
    [ypred, delta] = predict(mdl, X, 'Prediction', 'observation', 'Alpha', 0.05);
    lower_bounds(:, i) = ypred - delta(:, 1);
    upper_bounds(:, i) = ypred + delta(:, 2);
    
    fprintf('Multivariate regression for Trait %d completed.\n', i);
end

corr_matrix = corr(X);
figure;
% heatmap(corr_matrix, 'Colormap', 'cool', 'ColorScaling', 'log');
% title('Correlation Matrix Heatmap');
colormap('cool');

% Create the heatmap
heatmap(corr_matrix);
title('Correlation Matrix Heatmap');

% Display and save the predicted traits and prediction intervals
fprintf('Predicted Traits:\n');
disp(predicted_traits);
fprintf('Prediction Intervals (Lower and Upper Bounds):\n');
disp([lower_bounds, upper_bounds]);

% Complete the trait matrix by replacing missing values with predictions
complete_trait_matrix = data1;

figure;
for i = 1:size(Y, 2)
    subplot(2, ceil(size(Y, 2) / 2), i);
    histogram(predicted_traits(:, i));
    title(['Histogram of Predicted Trait ' num2str(i)]);
end

for i = 1:size(Y, 2)
    current_trait = Y(:, i);
    missing_indices = isnan(data1(:, i + 21));
    X_known = X(~missing_indices, :);
    Y_known = current_trait(~missing_indices);
    mdl = fitlm(X_known, Y_known);
    predicted_values = predict(mdl, X(missing_indices, :));
    complete_trait_matrix(missing_indices, i + 21) = predicted_values;
    
    fprintf('Missing value imputation for Trait %d completed.\n', i);
end

% Standardize the data
standardized_data = zscore(complete_trait_matrix);

% Apply PCA
[coeff, score, latent, ~, explained] = pca(standardized_data);

% Calculate cumulative variance explained
cumulative_explained = cumsum(explained);

% Choose the number of principal components to retain
desired_variance_explained = 95;
num_components_to_retain = find(cumulative_explained >= desired_variance_explained, 1);

% Retain the selected number of principal components
selected_components = coeff(:, 1:num_components_to_retain);
reduced_data = standardized_data * selected_components;

trait_index_to_visualize = 1; % Change to the desired trait index
figure;
plot(1:size(Y, 1), predicted_traits(:, trait_index_to_visualize), 'b', 'LineWidth', 1.5);
hold on;
plot(1:size(Y, 1), lower_bounds(:, trait_index_to_visualize), 'r--', 'LineWidth', 1);
plot(1:size(Y, 1), upper_bounds(:, trait_index_to_visualize), 'r--', 'LineWidth', 1);
title(['Prediction Intervals for Trait ' num2str(trait_index_to_visualize)]);
legend('Predicted', 'Lower Bound', 'Upper Bound');

fprintf('PCA completed.\n');

figure;
plot(1:length(cumulative_explained), cumulative_explained, 'go-', 'LineWidth', 1.5);
title('Cumulative Variance Explained by Principal Components');
xlabel('Number of Principal Components');
ylabel('Cumulative Explained Variance (%)');

% 'reduced_data' now contains the data reduced to the selected number of principal components
