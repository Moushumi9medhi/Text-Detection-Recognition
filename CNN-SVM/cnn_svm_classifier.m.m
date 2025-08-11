% ------------------------------------------------------------
% Purpose:
%   Train and evaluate a CNN + SVM hybrid classifier for 
%   distinguishing between "icon" and "text" images.
%
% Workflow:
%   1. Load dataset from folder structure.
%   2. Balance classes and split into training/testing sets.
%   3. Extract deep features using a pre-trained CNN.
%   4. Train a linear SVM classifier.
%   5. Evaluate accuracy on test set.
%
% Requirements:
%   - MATLAB R2016a or later
%   - Computer Vision Toolbox
%   - Pre-trained MatConvNet model (imagenet-caffe-alex.mat)
%
% Author: Adapted for project use
% ------------------------------------------------------------

close all;
clc;

%% 1. Load and Organize Dataset
datasetDir = fullfile('E:', 'iitkgp_project', 'Cnn_Svm_matlab');
classLabels = {'icons', 'texts'};

imds = imageDatastore(fullfile(datasetDir, classLabels), ...
    'LabelSource', 'foldernames');

% Display class counts
labelCount = countEachLabel(imds)

% Balance dataset (equal samples per class)
minClassCount = min(labelCount{:,2});
imds = splitEachLabel(imds, minClassCount, 'randomize');

disp('Balanced dataset:');
countEachLabel(imds)

% Show one example image from each class
iconIdx  = find(imds.Labels == 'icons', 1);
textIdx  = find(imds.Labels == 'texts', 1);

figure
subplot(1,2,1); imshow(imds.Files{iconIdx}); title('Example: Icon');
subplot(1,2,2); imshow(imds.Files{textIdx}); title('Example: Text');

%% 2. Load Pre-trained CNN (AlexNet in MatConvNet format)
cnnModelPath = fullfile(datasetDir, 'imagenet-caffe-alex.mat');
cnnModel = helperImportMatConvNet(cnnModelPath);

disp(cnnModel.Layers(1));       % First layer info
disp(cnnModel.Layers(end));     % Final classification layer

%% 3. Pre-processing Function for CNN Input
imds.ReadFcn = @(filename) preprocessImage(filename);

    function imgOut = preprocessImage(fname)
        img = imread(fname);
        if ismatrix(img) % Convert grayscale to RGB
            img = cat(3, img, img, img);
        end
        imgOut = imresize(img, [227 227]); % AlexNet input size
    end

%% 4. Train-Test Split (70% Train, 30% Test)
[trainSet, testSet] = splitEachLabel(imds, 0.70, 'randomize');

%% 5. Extract CNN Features from Training Data
featureLayer = 'fc7'; % Fully connected layer before classification
trainFeatures = activations(cnnModel, trainSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');

%% 6. Train Linear SVM on CNN Features
trainLabels = trainSet.Labels;
svmModel = fitcecoc(trainFeatures, trainLabels, ...
    'Learners', 'Linear', ...
    'Coding', 'onevsall', ...
    'ObservationsIn', 'columns');

%% 7. Evaluate on Test Set
testFeatures = activations(cnnModel, testSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
predictedLabels = predict(svmModel, testFeatures);

% Confusion matrix (normalized by row)
testLabels = testSet.Labels;
confMat = confusionmat(testLabels, predictedLabels);
confMat = bsxfun(@rdivide, confMat, sum(confMat, 2));
disp('Normalized Confusion Matrix:');
disp(confMat);

% Mean classification accuracy
meanAcc = mean(diag(confMat)) * 100;
fprintf('Mean Accuracy: %.2f%%\n', meanAcc);

%% 8. Manual Test on Individual Images
correctPredictions = 0;
numTestImages = numel(testSet.Files);

for idx = 1:numTestImages
    img = preprocessImage(testSet.Files{idx});
    imgFeatures = activations(cnnModel, img, featureLayer, 'OutputAs', 'columns');
    predicted = predict(svmModel, imgFeatures);

    if predicted == testLabels(idx)
        correctPredictions = correctPredictions + 1;
    end
end

finalAccuracy = (correctPredictions / numTestImages) * 100;
fprintf('Final Accuracy on Test Images: %.2f%%\n', finalAccuracy);


