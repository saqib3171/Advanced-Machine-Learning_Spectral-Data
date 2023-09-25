clc;
close all;
clear all;
warning off;

%% Week 2

% Define the filenames for data files
filename = 'data_part_1.csv';
filename1 = 'data_part_2.xlsx';

% Read data from CSV and Excel files into variables
data = readmatrix(filename);
data1 = xlsread(filename1);

% Select traits from dataset1 for the boxplot
selected_variables = data(:, 2:21);
% Define variable names
variable_names = {'Anthocyanin content','Boron content', 'C content', 'Ca content', 'Carotenoid content', 'Cellulose', 'Chl content ','Copper content', 'EWT', 'Fiber', 'LAI', 'LMA', 'Lignin','Magnesium content', 'Manganese content', 'N content', 'NSC', 'Phosphorus content', 'Potassium content', 'Sulfur content'};
% Create a figure for the boxplot
figure;
% Create a boxplot for the selected traits
boxplot(selected_variables, 'Colors', 'bgrm','Notch', 'on', 'OutlierSize', 5,'Labels', variable_names);

xlabel('Variables (Predictors)');
ylabel('Values');
title('Boxplot of Variables from dataset1');
% Rotate x-axis labels for better visibility
xtickangle(45);

% Adjusting the figure size
set(gcf, 'Position', [100, 100, 800, 600]);
%saveas(gca,'f1.png')

% Select traits from dataset2 for the boxplot
selected_variables2 = data1(:, 2:38);

% Define variable names 
variable_names2 = {'Anthocyanin content','Anthocyanin concentration', 'Boron content','Boron concentration ', 'C content', 'C concentration', 'Ca content','Ca concentration','Carotenoid content', 'Carotenoid concentration', 'Cellulose (mg/cmÂ²)', 'Cellulose (mg/g)','Chl content (Î¼g/cmÂ²)','Chlorophyll concentration (mg/g)', 'Copper content (mg/cmÂ²)','Copper concentration (mg/g)', 'EWT (mg/cmÂ²)', 'Fiber (mg/cmÂ²)', 'Fiber (mg/g)','LAI (mÂ²/mÂ²)', 'LMA (g/mÂ²)', 'Lignin (mg/cmÂ²)', 'Lignin (mg/g)', 'Magnesium content (mg/cmÂ²)', 'Magnesium concentration (mg/g)', 'Manganese content (mg/cmÂ²)','Manganese concentration (mg/g)', 'N content (mg/cmÂ²)', 'N concentration (mg/g)', 'NSC (mg/cmÂ²)','NSC (mg/g)', 'Phosphorus content (mg/cmÂ²)', 'P concentration (mg/g)', 'Potassium content (mg/cmÂ²)','Potassium concentration (mg/g)', 'Sulfur content (mg/cmÂ²)','Sulfur concentration (mg/g)'};

% Create a figure for the boxplot
figure;
% Create a boxplot for the selected variables
boxplot(selected_variables2, 'Colors', 'bgrm','Notch', 'on', 'OutlierSize', 5,'Labels', variable_names2);
xlabel('Variables (Predictors)');
ylabel('Values');
title('Boxplot of Variables from dataset2');

% Rotate x-axis labels for better visibility
xtickangle(45);

% Adjust the figure size 
set(gcf, 'Position', [100, 100, 800, 600]);
saveas(gca,'f2.png')

% Check for missing values in dataset1
missing_values = isnan(data);
% Check for missing values in dataset2
missing_values1 = isnan(data1); 

% Display the missing values in dataset1
fprintf('Missing values in the dataset1:\n');
disp(missing_values);
% Display the missing values in dataset2
fprintf('Missing values in the dataset2:\n');
disp(missing_values1);

% Create a figure with two subplots
figure
subplot(1,2,1)
% Plot a surface plot of a subset of 'data'
surf(data(2:50,22:end))
subplot(1,2,2)

% Plot an image of a subset of 'data'
imagesc(data(2:50,22:end))
colormap hot

% display an image of a subset of 'data1'
figure
imagesc(data1(2:50,31:end))
colormap hot

% Initialize an empty array for FFT results
fftarr = [];

% Display FFT plots
figure
hold on
% Iterate through rows from 100 to 120
for i = 100:120
    % Compute the FFT of a row in 'data' and calculate its magnitude
    p = abs(fftshift(fft(data(i,22:end))));
    % Plot the FFT result
    plot(p)
    % Pause briefly for visualization
    pause(0.1)
    % Append the FFT result to 'fftarr'
    fftarr = [fftarr; p];
end
figure % carbon 
plot(data(2:end,4))

figure % cellulose
plot(data(2:end,6))

%% Week 3

% Detect Outliers in dataset1 using Z-Scores
z_threshold = 0.03; % Adjust this threshold as needed
z_scores = zscore(data(:,:));
outliers_data = abs(z_scores) > z_threshold;
surf(z_scores)
%saveas(gca,'fz.png')
z_scores(z_scores>=1)=1
[r,c]=find(z_scores)
figure
plot(r,c,'*')
saveas(gca,'fp.png')
% Detect Outliers in dataset2 using Z-Scores
z_threshold_1 = 0.03; % Adjust this threshold as needed
z_scores_1 = zscore(data1(:,:));
outliers_data_1 = abs(z_scores_1) > z_threshold_1;
surf(z_scores_1)
%saveas(gca,'fz1.png') 
z_scores_1(z_scores_1>=1)=1
[r1,c1]=find(z_scores_1)
figure
plot(r1,c1,'*')

% Define the variables for the multivariate regression
% X represents the independent variables (known traits)
% Y represents the dependent variables (traits to predict)
X = data(:, 2:21); % Independent variables (known traits)
Y = data(:, 22:end); % Dependent variables (traits to predict)

% Initialize matrices to store predictions and prediction intervals
predicted_traits = NaN(size(Y)); % Matrix for predicted values
prediction_intervals = NaN(size(Y, 1), size(Y, 2), 2); % Matrix for prediction intervals (lower and upper bounds)

% Loop through each trait to predict
for i = 1:size(Y, 2)
    % Select the current trait for prediction
    current_trait = Y(:, i);

    % Find indices where the current trait has missing data
    missing_indices = isnan(current_trait);

    % Create a model for the current trait using known data
    X_known = X(~missing_indices, :);
    Y_known = current_trait(~missing_indices);

    % Fit a simple linear regression model
    mdl = fitlm(X_known, Y_known);

    % Predict the current trait for all data points
    predicted_values = predict(mdl, X);

    % Store predicted values in the result matrix
    predicted_traits(:, i) = predicted_values;

    % Calculate prediction intervals
    [ypred, delta] = predict(mdl, X, 'Prediction', 'observation', 'Alpha', 0.05);

    % Store lower and upper bounds separately
    lower_bounds(:, i) = ypred - delta(:, 1);
    upper_bounds(:, i) = ypred + delta(:, 2);
end

% Display and save the predicted traits and prediction intervals
disp('Predicted Traits:');
disp(predicted_traits);

disp('Prediction Intervals (Lower and Upper Bounds):');
disp(lower_bounds);
disp(upper_bounds);

% Now combine the predicted traits with observed data to create a complete trait matrix.
% We can replace the missing values in original data with the predicted values if needed.
% Initialize a complete_trait_matrix with NaN values
complete_trait_matrix = data1; % Start with the original data1

% Loop through each trait to predict
for i = 1:size(Y, 2)
    % Select the current trait for prediction
    current_trait = Y(:, i);

    % Find indices where the current trait has missing data in dataset1
    missing_indices = isnan(data1(:, i + 21)); % Assuming traits start from column 22

    % Create a model for the current trait using known data
    X_known = X(~missing_indices, :);
    Y_known = current_trait(~missing_indices);

    % Fit a simple linear regression model
    mdl = fitlm(X_known, Y_known);

    % Predict the current trait for missing data points in dataset1
    predicted_values = predict(mdl, X(missing_indices, :));

    % Replace missing values in complete_trait_matrix with predicted values
    complete_trait_matrix(missing_indices, i + 21) = predicted_values;
end

% Now, complete_trait_matrix contains the original data with missing values replaced by predicted values.

% Standardize the data
standardized_data = zscore(complete_trait_matrix);

% Apply PCA
[coeff, score, latent, ~, explained] = pca(standardized_data);

% 'coeff' contains the principal component loadings
% 'score' contains the scores (projected data)
% 'latent' contains the eigenvalues (variance explained by each component)
% 'explained' contains the percentage of variance explained by each component

% Plot the explained variance
figure;
plot(explained, 'bo-');
xlabel('Principal Component');
ylabel('Percentage of Variance Explained');
title('Explained Variance by Principal Components');

% Choose the number of principal components to retain based on your desired explained variance threshold
% For example, you can retain components that collectively explain 95% of the variance
desired_variance_explained = 95; 

cumulative_explained = cumsum(explained);
num_components_to_retain = find(cumulative_explained >= desired_variance_explained, 1);

% Retain the selected number of principal components
selected_components = coeff(:, 1:num_components_to_retain);
reduced_data = standardized_data * selected_components;

% 'reduced_data' now contains the data reduced to the selected number of principal components
