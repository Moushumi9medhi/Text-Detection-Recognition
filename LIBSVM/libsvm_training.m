close all;
opengl software;


addpath('E:/iitkgp_project/histogram/LIBSVM/libsvm-3.21/matlab');
addpath('E:/iitkgp_project/histogram/LIBSVM/libsvm-3.21');


trainFile = 'path-to-Scaled_LIBSVM_trainingfile_histogram.train';
testFile  = 'path-to-Scaled_LIBSVM_testingfile_histogram.test';


[trainLabels, trainFeatures] = libsvmread(trainFile);
[testLabels, testFeatures]   = libsvmread(testFile);


svmParams = '-s 0 -t 2 -c 2 -g 8';  % C=2, gamma=8, RBF kernel
svmModel  = svmtrain(trainLabels, trainFeatures, svmParams);


numTestSamples     = size(testFeatures, 1);
misclassifiedIdx   = [];
errorCounter       = 0;

for rowIdx = 1:numTestSamples
    % Extract single test sample
    sampleData  = testFeatures(rowIdx, :);
    sampleLabel = testLabels(rowIdx);

    % Run prediction
    [predLabel, ~, ~] = svmpredict(sampleLabel, sampleData, svmModel);

    % Track misclassifications
    if predLabel ~= sampleLabel
        errorCounter = errorCounter + 1;
        misclassifiedIdx(end+1) = rowIdx; %#ok<AGROW>
    end
end


disp('Misclassified sample indices:');
disp(misclassifiedIdx);
fprintf('Total misclassifications: %d out of %d samples.\n', ...
        errorCounter, numTestSamples);
