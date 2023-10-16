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

%% Check for missing values in dataset1 and dataset2
missing_values = isnan(data);
missing_values1 = isnan(data1);

fprintf('Missing values in dataset1:\n');
disp(missing_values);
fprintf('Missing values in dataset2:\n');
disp(missing_values1);

%% Select traits from dataset1 for the boxplot
selected_variables = data(:, 2:21);
variable_names = {'Anthocyanin content','Boron content', 'C content', 'Ca content', 'Carotenoid content', 'Cellulose', 'Chl content ','Copper content', 'EWT', 'Fiber', 'LAI', 'LMA', 'Lignin','Magnesium content', 'Manganese content', 'N content', 'NSC', 'Phosphorus content', 'Potassium content', 'Sulfur content'};

%% Create a figure for the boxplot
figure;
boxplot(selected_variables, 'Colors', 'bgrm','Notch', 'on', 'OutlierSize', 5,'Labels', variable_names);
xlabel('Variables (Predictors)');
ylabel('Values');
title('Boxplot of Variables from dataset1');
xtickangle(45);
set(gcf, 'Position', [100, 100, 800, 600]);
legend(variable_names);

%% Select traits from dataset2 for the boxplot
selected_variables2 = data1(:, 2:38);
variable_names2 = {'Anthocyanin content','Anthocyanin concentration', 'Boron content','Boron concentration ', 'C content', 'C concentration', 'Ca content','Ca concentration','Carotenoid content', 'Carotenoid concentration', 'Cellulose (mg/cm)', 'Cellulose (mg/g)','Chl content (g/cm)','Chlorophyll concentration (mg/g)', 'Copper content (mg/cm)','Copper concentration (mg/g)', 'EWT (mg/cm)','Fiber (mg/cm)','Fiber (mg/g)','LAI (m/m)','LMA (g/m)','Lignin (mg/cm)','Lignin (mg/g)', 'Magnesium content (mg/cm)','Magnesium concentration (mg/g)', 'Manganese content (mg/cm)','Manganese concentration (mg/g)', 'N content (mg/cm)','N concentration (mg/g)', 'NSC (mg/cm)','NSC (mg/g)', 'Phosphorus content (mg/cm)','P concentration (mg/g)', 'Potassium content (mg/cm)','Potassium concentration (mg/g)', 'Sulfur content (mg/cm)','Sulfur concentration (mg/g)'};

%% Create a figure for the boxplot
figure;
boxplot(selected_variables2, 'Colors', 'bgrm','Notch', 'on', 'OutlierSize', 5,'Labels', variable_names2);
xlabel('Variables (Predictors)');
ylabel('Values');
title('Boxplot of Variables from dataset2');
xtickangle(45);
set(gcf, 'Position', [100, 100, 800, 600]);
legend(variable_names2);

%% fft
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

%% %% Principal Component Analysis (PCA)
% Standardize the data (mean = 0, standard deviation = 1)
Z = zscore(data(:, 2:21)); % Using data from dataset1

% Perform PCA
[coeff, score, latent, ~, explained] = pca(Z);

% Plot the explained variance
figure;
bar(explained);
xlabel('Principal Component');
ylabel('Explained Variance (%)');
title('Explained Variance by Principal Component');

% Choose the number of components to retain (e.g., 2)
num_components_to_retain = 2;

% Retain the top 'num_components_to_retain' principal components
X_pca = score(:, 1:num_components_to_retain);

% Display the retained PCA components
disp('Retained PCA Components:');
disp(coeff(:, 1:num_components_to_retain));
%% % Calculate the correlation matrix for the retained PCA components
correlation_matrix = corr(X_pca);

% Display the correlation matrix
disp('Correlation Matrix of Retained PCA Components:');
disp(correlation_matrix);

% Create a heatmap of the correlation matrix with a custom colormap
figure;
heatmap(correlation_matrix, 'Colormap', jet(256), 'ColorScaling', 'scaled', 'XData', variable_names(1:num_components_to_retain), 'YData', variable_names(1:num_components_to_retain));
title('Correlation Matrix Heatmap of Retained PCA Components');

%% Surface plot of dataset1
figure;
surf(data(2:50,22:end));
xlabel('X-axis');
ylabel('Y-axis');
zlabel('Z-axis');
title('Surface Plot of dataset1');

%% Image plot of dataset1
figure;
imagesc(data(2:50,22:end));
xlabel('X-axis');
ylabel('Y-axis');
title('Image Plot of dataset1');
colormap hot;
set(gcf, 'Position', [100, 100, 800, 600]);

%% Image plot of dataset2
figure;
imagesc(data1(2:50,22:end));
xlabel('X-axis');
ylabel('Y-axis');
title('Image Plot of dataset2');
%% 
%% Robust prediction intervals for multivariate regression
X = data(:, 2:21); % Independent variables (known traits)
Y = data(:, 22:end); % Dependent variables (traits to predict)

%% Initialize matrices for predictions and prediction intervals
predicted_traits = NaN(size(Y));
prediction_intervals = NaN(size(Y, 1), size(Y, 2), 2);

%% Loop through each trait to predict
for i = 1:size(Y, 2)
    current_trait = Y(:, i);
    missing_indices = isnan(current_trait);
    X_known = X(~missing_indices, :);
    Y_known = current_trait(~missing_indices);

    %% Perform multivariate regression
    mdl = fitlm(X_known, Y_known);

    %% Predict the current trait for all data points
    predicted_values = predict(mdl, X);
    predicted_traits(:, i) = predicted_values;

    %% Calculate prediction intervals
    %% Calculate prediction intervals
[ypred, delta] = predict(mdl, X, 'Prediction', 'observation', 'Alpha', 0.05);
lower_bounds(:, i) = ypred - delta(:, 1);
upper_bounds(:, i) = ypred + delta(:, 2);


    fprintf('Multivariate regression for Trait %d completed.\n', i);
end

%% Plot the predicted trait and prediction intervals for the first trait
figure;
plot(1:size(Y, 1), predicted_traits(:, 1), 'b', 'LineWidth', 1.5);
hold on;
plot(1:size(Y, 1), lower_bounds(:, 1), 'r--', 'LineWidth', 1);
plot(1:size(Y, 1), upper_bounds(:, 1), 'r--', 'LineWidth', 1);
title('Predicted Trait and Prediction Intervals for Trait 1');
legend('Predicted', 'Lower Bound', 'Upper Bound');

