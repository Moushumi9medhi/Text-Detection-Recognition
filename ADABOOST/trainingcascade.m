%% trainingcascade.m
% ------------------------------------------------------------
% Purpose:
%   Train a text detector using MATLAB's AdaBoost cascade 
%   classifier with general positive and negative samples.
%
% Requirements:
%   - MATLAB Computer Vision Toolbox
%   - Large positive image set (~1000+ images recommended)
%   - Negative image folder with non-text samples
%
% Output:
%   - Trained classifier saved as TEXTDetector.xml
%
% Year: 2017
% ------------------------------------------------------------

opengl software;
positiveinstancesTEXT = struct;

% Prepare positive samples
for i = 1:1156
    fileName = strcat('./path-to-positive-image-folder/', num2str(i), '.jpg');
    img = imread(fileName);
    [height, width, ~] = size(img);
    clear img;
    positiveinstancesTEXT(i).imageFilename = fileName;
    positiveinstancesTEXT(i).objectBoundingBoxes = [1, 1, width, height];
end

% Train cascade classifier
trainCascadeObjectDetector(...
    'TEXTDetector.xml', ...
    positiveinstancesTEXT, ...
    './path-to-negative-image-folder', ...
    'FalseAlarmRate', 0.5, ...
    'NumCascadeStages', 15);

% ------------------------------------------------------------
% Training Tips:
%  - Increase stages for fewer false positives
%  - Each stage uses negatives from false positives of previous stage
%  - Too many stages can overfit if positives are insufficient
% ------------------------------------------------------------
