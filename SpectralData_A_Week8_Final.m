clc; clear all; close all;
warning off;

%% Define the filenames for data files
filename = 'data_part_1.csv';
filename1 = 'data_part_2.xlsx';

%% Read data from CSV and Excel files into variables
data1 = csvread(filename, 1, 1); % Skip the header row and first column
data2 = xlsread(filename1);

%% Check if there are missing values in dataset1 and dataset2
missing_values_data1 = any(isnan(data1), 'all');
missing_values_data2 = any(isnan(data2), 'all');

if missing_values_data1
    fprintf('Dataset1 has missing values.\n');
else
    fprintf('Dataset1 has no missing values.\n');
end

if missing_values_data2
    fprintf('Dataset2 has missing values.\n');
else
    fprintf('Dataset2 has no missing values.\n');
end

%% Select traits from dataset1 for the boxplot
selected_variables = data1(:, 2:21);
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

%% Define the variable names for the content columns
variable_names2 = {'Anthocyanin content', 'Boron content', 'C content', 'Ca content', 'Carotenoid content', 'Cellulose', 'Chl content', 'Copper content', 'EWT', 'Fiber', 'LAI', 'LMA', 'Lignin', 'Magnesium content', 'Manganese content', 'N content', 'NSC', 'Phosphorus content', 'Potassium content', 'Sulfur content'};
%% Extract only the content columns from the dataset
selected_columns_indices = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 20, 21, 23, 25, 27, 29, 31, 33, 35, 37];
selected_variables2 = data2(:, selected_columns_indices);
%% Create a figure for the boxplot
figure;
boxplot(selected_variables2, 'Colors', 'bgrm','Notch', 'on', 'OutlierSize', 5,'Labels', variable_names2);
xlabel('Variables (Predictors)');
ylabel('Values');
title('Boxplot of Variables from dataset2');
xtickangle(45);
set(gcf, 'Position', [100, 100, 800, 600]);
legend(variable_names2);

%% Define X and Y
X = data1(:, 22:end); % Wavelengths 
Y = data1(:, 2:21); % Traits 

%% Visualize correlations between traits in Y
figure;
corr_matrix = corr(Y, 'rows', 'complete'); % Correlation matrix
heatmap(corr_matrix);
title('Correlation Heatmap of Traits');

%% plot for a subset of traits (Kept first 5 traits only)
figure;
plotmatrix(data1(:, 1:5));
title('Plot Matrix of First 5 Traits');

%% Visualize distribution of the first trait
figure;
histogram(data2(:, 2), 50);
title('Histogram of the First Trait');
xlabel('Trait Values');
ylabel('Frequency');

%% Split data into 80% training and 20% test sets
c = cvpartition(size(X,1), 'Holdout', 0.2);  % Partition data 

X_train = X(c.training, :);
Y_train = Y(c.training, :);
X_test = X(c.test, :);
Y_test = Y(c.test, :);


%% Center and scale the data based on the training set 
mean_train = mean(Y_train);
std_train = std(Y_train);

Y_train = (Y_train - mean_train) ./ std_train;
Y_test = (Y_test - mean_train) ./ std_train;

%% Apply K-fold cross-validation on the training data
num_traits = 20; % Number of traits 
num_components_range = 1:20; % Range of PLS components to evaluate from 1 to 20
K = 10;    % K-fold cross-validation to 10
coeff_pls_all = cell(1, num_traits); % store PLS coefficients for each trait

%% Initialize arrays to store RMSE and R-squared values for each trait
RMSE_values = zeros(1, num_traits);
R2_values = zeros(1, num_traits);

%% PLS Regression

% Loop through each trait to perform PLS regression and cross-validation.
for trait_num = 1:num_traits 

current_trait_train = Y_train(:, trait_num);
missing_indices_train = isnan(current_trait_train);
X_known_train = X_train(~missing_indices_train, :);
Y_known_train = current_trait_train(~missing_indices_train);
    
% Initialize error vectors for storing cross-validation errors.
cv_error = zeros(1, length(num_components_range));
  
% Evaluate PLS regression performance for different numbers of components.
for j = 1:20
num_components = j;
cv_error_temp = zeros(1, K);
        
% Set up K-fold cross-validation on the known training data.
cv = cvpartition(length(Y_known_train), 'KFold', K);
        for k = 1:K
            X_train_cv = X_known_train(training(cv, k), :);
            Y_train_cv = Y_known_train(training(cv, k));
            X_val_cv = X_known_train(test(cv, k), :);
            Y_val_cv = Y_known_train(test(cv, k));
     
            % Standardize the training and validation data.
            mean_train_cv = mean(X_train_cv);
            std_train_cv = std(X_train_cv);
            X_train_cv = (X_train_cv - mean_train_cv) ./ std_train_cv;
            X_val_cv = (X_val_cv - mean_train_cv) ./ std_train_cv;
            
            % Perform PLS regression on the training data of the current fold.
            [~, ~, ~, ~, BETA] = plsregress(X_train_cv, Y_train_cv, num_components);
            
            % Predict the validation data using the obtained model.
            Y_pred_cv = [ones(size(X_val_cv,1),1) X_val_cv] * BETA;

            % Calculate the mean squared error for the current fold.
            cv_error_temp(k) = mean((Y_val_cv - Y_pred_cv).^2);
            
            % Calculate the Q2 statistic for model evaluation.
            TSS = sum((Y_train_cv - mean(Y_train_cv)).^2);
            PRESS = sum((Y_val_cv - Y_pred_cv).^2);
            Q2(k) = 1 - PRESS/TSS;
        end
% Compute the average Q2 and MSE across all folds.
cv_q2(j)    = mean(Q2);
cv_error(j) = mean(cv_error_temp);
    end
     
    % Determine the optimal number of components with a criterion based on Q2 values.
    [~, idxComp] = min(round(cv_q2, 2) - round((max(round(cv_q2, 2)) - 0.03),2));
    optimal_num_components = find(round(cv_q2, 2) == round((max(round(cv_q2, 2)) - 0.03),2), 1);
    fprintf('For Trait %d, optimal number of PLS components: %d\n', trait_num, optimal_num_components);

    % Perform PLS regression with the optimal number of components to obtain the final model.
    try
        [~, ~, ~, ~, BETA_optimal] = plsregress(X_known_train, Y_known_train, optimal_num_components);
    catch
        
        [~, ~, ~, ~, BETA_optimal] = plsregress(X_known_train, Y_known_train, 7);
    end

    
%% Predict on the test data
  predicted_values_test = [ones(size(X_test,1),1) X_test] * BETA_optimal;
  predicted_traits_test(:, trait_num) = predicted_values_test;
    
%% Fill in missing traits in the original data

missing_indices_trait = isnan(Y(:, trait_num));  % Identify the indices of missing values 

% Predict the missing values using the optimal PLS model coefficients (BETA_optimal).
predicted_missing_values = [ones(sum(missing_indices_trait),1) X(missing_indices_trait, :)] * BETA_optimal;

% Fill in the missing values in the dataset with the predicted values.
Y(missing_indices_trait, trait_num) = predicted_missing_values;

% Identify the indices of known (non-missing) values for the current trait in the test set.
known_indices_test = ~isnan(Y_test(:, trait_num));

Y_actual_test = Y_test(known_indices_test, trait_num); % Extract the actual known values of the trait from the test set.

Y_pred_test = predicted_traits_test(known_indices_test, trait_num); 

%% Calculate RMSE on test data

RMSE_test = sqrt(mean((Y_actual_test - Y_pred_test).^2));  % Calculate the RMSE 

RMSE_values(trait_num) = RMSE_test;   % Store the calculated RMSE value 

fprintf('Test RMSE for Trait %d: %f\n', trait_num, RMSE_test);  % Output the RMSE value for the current trait 

    
%% Calculate R-squared on test data
  
residuals_test = Y_actual_test - Y_pred_test;  % Calculate residuals 
ss_res_test = sum(residuals_test.^2);   % Compute the sum of squared residuals 

ss_tot_test = sum((Y_actual_test - mean(Y_actual_test)).^2);   % Compute the total sum of squares 
R2_test = 1 - (ss_res_test/ss_tot_test);  % Calculate the coefficient of determination (R^2)

R2_values(trait_num) = R2_test;

  fprintf('Test R-squared for Trait %d: %f\n', trait_num, R2_test);
end
%% Rank the traits based on RMSE and R-squared values
[~, RMSE_rank] = sort(RMSE_values);
[~, R2_rank] = sort(R2_values, 'descend');

fprintf('\nRanking based on RMSE:\n');
if length(RMSE_values) == num_traits && length(R2_values) == num_traits
    fprintf('\nRanking based on RMSE:\n');
    for i = 1:num_traits
        fprintf('Rank %d: Trait %d\n', i, RMSE_rank(i));
    end

    fprintf('\nRanking based on R-squared:\n');
    for i = 1:num_traits
        fprintf('Rank %d: Trait %d\n', i, R2_rank(i));
    end
else
    fprintf('Error: RMSE_values or R2_values not populated correctly.\n');
end

%% Apply PCA on the entire trait matrix
Z = zscore(Y);
[coeff_pca, score, latent, tsquared, explained] = pca(Z);


%% PCA Residual Analysis

% Number of PCs to consider for reconstruction
k = 6; 

% Reconstruct data matrix using the first k PCs
Z_reconstructed = score(:, 1:k) * coeff_pca(:, 1:k)';

% Compute the residuals matrix E
E = Z - Z_reconstructed;

% Compute the summed residuals for each variable
Res = sum(abs(E));

% Plot a bar plot for the summed residuals
figure;

if length(Res) == length(variable_names)
    bar(categorical(variable_names), Res);
else
    bar(1:length(Res), Res);
    currentLabels = xticklabels();
newLabels = cell(size(currentLabels));
for i = 1:length(currentLabels)
    newLabels{i} = currentLabels{i};
end
for i = length(variable_names)+1:length(Res)
    newLabels{i} = ['Trait ' num2str(i)];
end

end

title('Summed Residuals for Each Variable');
xlabel('Variables');
ylabel('Summed Residuals');
xtickangle(45);

num_PCs = 6;

%% Display variance explained by the first 5-6 PCs
for i = 1:num_PCs
    fprintf('PC%d explains %.2f%% of the total variance.\n', i, explained(i));
end

%% Cumulative variance explained by the first 5-6 PCs
cumulative_explained = cumsum(explained);
fprintf('\nThe first %d PCs together explain %.2f%% of the total variance.\n', num_PCs, cumulative_explained(num_PCs));

%% Compute T2 

% Number of observations and variables
[n, p] = size(Z);

% Compute the T2  for each observation
T2 = zeros(n,1);
for i = 1:n
    T2(i) = n * score(i,:) * inv(diag(latent)) * score(i,:)';
end

% Plot T2 values
figure;
plot(T2, 'o-');
title('T2 for Each Observation');
xlabel('Observation');
ylabel('T2 Value');

%% This code can be used for visualization
% % Bar plots for Loadings for the first 5-6 PCs
% for i = 1:num_PCs
%     figure;
%     bar(coeff_pca(:,i));
%     title(['Loadings for PC' num2str(i)]);
%     xlabel('Variables');
%     ylabel('Coefficient values');
%     xticks(1:length(variable_names));
%     xticklabels(variable_names);
%     xtickangle(45);
% end
%% ///////////////////////////

% Number of PCs to plot
num_PCs_to_plot = min(6, num_PCs); % Take the smaller value between 6 and num_PCs
figure;

% Loop to generate subplots for each PC
for i = 1:num_PCs_to_plot
    % Create a subplot
    subplot(ceil(num_PCs_to_plot/2), 2, i); % Assuming 2 columns of subplots
    bar(coeff_pca(:,i));
    title(['Loadings for PC' num2str(i)]);
    xlabel('Variables');
    ylabel('Coefficient values');
    xticks(1:length(variable_names));
    xticklabels(variable_names);
    xtickangle(45);
end
%tight_layout();

% Scatter plots for the scores of the first 5-6 PCs
for i = 1:num_PCs-1
    figure;
    scatter(score(:,i), score(:,i+1));
    title(['Score plot for PC' num2str(i) ' and PC' num2str(i+1)]);
    xlabel(['PC' num2str(i)]);
    ylabel(['PC' num2str(i+1)]);
end

%% Variance Explained Plot
figure;
plot(cumsum(explained), '-o');
title('Cumulative Variance Explained');
xlabel('Number of Principal Components');
ylabel('Total Variance Explained (%)');

%% Biplot for the first two principal components with variable names
num_additional_vars = size(coeff_pca, 1) - length(variable_names);
for i = 1:num_additional_vars
    variable_names{end+1} = ['Var' num2str(length(variable_names)+i)];
end

figure;
biplot(coeff_pca(:,1:2), 'Scores', score(:,1:2), 'VarLabels', variable_names);
title(['Biplot for First Two Principal Components (PC1: ' num2str(explained(1)) '%, PC2: ' num2str(explained(2)) '%)']);

%% Loading plot
figure;
plot(coeff_pca(:,1), coeff_pca(:,2), 'r*');
xlabel('1st Principal Component');
ylabel('2nd Principal Component');
title('Loading Plot');

%% Identify the worst performing trait based on RMSE
[~, worst_RMSE_idx] = max(RMSE_values);

% Print out the worst-performing trait index
fprintf('Worst performing trait based on RMSE is Trait %d\n', worst_RMSE_idx);

%% Extract the wavelengths for the worst performing trait from the calibration partition

worst_trait_wavelengths = X_train(~isnan(Y_train(:, worst_RMSE_idx)), :);

%% Perform PCA on the wavelengths for the worst performing trait
[coeff_pca_worst, score_pca_worst, latent_pca_worst, tsquared_pca_worst, explained_pca_worst] = pca(worst_trait_wavelengths);

%% Visualize the explained variance by the first few principal components
figure;
plot(cumsum(explained_pca_worst), '-o');
title('Cumulative Variance Explained for Worst Performing Trait');
xlabel('Number of Principal Components');
ylabel('Total Variance Explained (%)');

