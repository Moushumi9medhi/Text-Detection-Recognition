%% cascadeclassifierTEXT.m
% ------------------------------------------------------------
% Purpose:
%   Detects text regions in an image using a trained AdaBoost
%   cascade classifier (trained separately using MATLAB's 
%   trainCascadeObjectDetector).
%
% Requirements:
%   - MATLAB Computer Vision Toolbox
%   - Trained classifier XML file (TEXTDetector.xml)
%
% Year: 2017
% ------------------------------------------------------------

close all;
opengl software;

% Load the pre-trained AdaBoost cascade classifier
detector = vision.CascadeObjectDetector('TEXTDetector.xml');

% Read the test image
img = imread('C:\Users\Asus\Downloads\1.png');

% Detect text bounding boxes
bbox = step(detector, img);

% Annotate detected regions on the image
detectedImg = insertObjectAnnotation(img, 'rectangle', bbox, 'text');

% Display the annotated image
figure;
imshow(detectedImg);
title('Detected Text Regions');

% ------------------------------------------------------------
% Notes:
%  - Increase the number of stages in training for more robust detection
%  - Tune 'MinSize' and 'MergeThreshold' properties of the detector for better results
% ------------------------------------------------------------
