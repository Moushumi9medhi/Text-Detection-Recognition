close all;
opengl software;

imageDir = 'path-to-training-set/';
imageFiles = dir(fullfile(imageDir, '*.jpg'));
numImages = numel(imageFiles);


histFeatures = zeros(numImages, 16);


for idx = 1:numImages
    imgPath = fullfile(imageDir, imageFiles(idx).name);
    rgbImg = imread(imgPath);
    grayImg = rgb2gray(rgbImg);
    normImg = mat2gray(grayImg);
    [binCounts, ~] = imhist(normImg, 16);
    histFeatures(idx, :) = binCounts.';
end

first_cl_=....% Number of images in the first class
classLabels = zeros(numImages, 1);
classLabels(1:first_cl_) = -1;
classLabels(first_cl_+1:end) = 1;

trainingMatrix = [classLabels, histFeatures];
csvPath = 'training_histogram_data.csv';
csvwrite(csvPath, trainingMatrix);

dataFromCSV = csvread(csvPath);
labelsFromCSV = dataFromCSV(:, 1);
featureMatrix = sparse(dataFromCSV(:, 2:end));

libsvmPath = 'LIBSVM_trainingfile_histogram.train';
libsvmwrite(libsvmPath, labelsFromCSV, featureMatrix);
