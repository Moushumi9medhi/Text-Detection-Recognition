% ------------------------------------------------------------
% Stroke Width Transform (SWT) for Text Detection
%
%
% Requirements:
%   - MATLAB with Image Processing Toolbox
%   - swtmex MEX file in MATLAB path
%
% ------------------------------------------------------------

close all;
opengl software;

%% Parameters
searchDirection = 1;   % 1 for dark-on-light, -1 for light-on-dark text
maxStrokeWidth = 20;   % Maximum stroke width (in pixels)

%% Load and Display Original Image
inputImage = imread('h11.png');
figure; imshow(inputImage); title('Original Image');

%% Convert to Grayscale (if RGB)
if size(inputImage, 3) == 3
    grayImage = rgb2gray(inputImage);
else
    grayImage = inputImage;
end

%% Convert to Single Precision
grayImage = im2single(grayImage);

%% Step 1: Edge Detection (Canny)
edgeMask = single(edge(grayImage, 'canny', 0.1));
figure; imshow(edgeMask); title('Canny Edge Map');

%% Step 2: Gaussian Smoothing
gaussianFilter = fspecial('gaussian', [3 3], 0.3 * (2.5 - 1) + 0.8);
smoothedImage = imfilter(grayImage, gaussianFilter);

%% Step 3: Gradient Computation (Prewitt)
gradX = imfilter(smoothedImage, fspecial('prewitt')');
gradY = imfilter(smoothedImage, fspecial('prewitt'));

% Median filtering to reduce noise
gradX = single(medfilt2(gradX, [3 3]));
gradY = single(medfilt2(gradY, [3 3]));

%% Step 4: Stroke Width Transform via MEX
[strokeWidthMap, connectedCompMap] = swtmex(edgeMask.', gradX.', gradY.', ...
                                            searchDirection, maxStrokeWidth);

% Transpose outputs back
strokeWidthMap = strokeWidthMap';
connectedCompMap = double(connectedCompMap');

%% Step 5: Display SWT Output
figure; imshow(strokeWidthMap, []); title('Stroke Width Transform Map');

