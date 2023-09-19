%% Project Work

clc;close all;clear all;warning off;


% Define the filenames for data files
filename = 'data_part_1.csv';
filename1 = 'data_part_2.xlsx';

% Read data from CSV and Excel files into variables
data = readmatrix(filename); 
data1 = readmatrix(filename1); 

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

