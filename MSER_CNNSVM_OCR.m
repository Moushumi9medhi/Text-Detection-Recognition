%% Stage 1: Load and prepare image
close all;
origImg = imread('G:\part1\IMG.png');
origImg = imresize(origImg, 2);
grayImg = rgb2gray(origImg);

%% Stage 2: MSER detection
mserCandidates = detectMSERFeatures(grayImg, ...
    'RegionAreaRange', [1 floor(0.0005 * numel(grayImg))], ...
    'ThresholdDelta', 3);

[mserStats, mserConn] = regionprops(mserCandidates.Connectivity, ...
    'BoundingBox', 'Eccentricity', 'Solidity', 'Extent', 'Euler', ...
    'Image', 'ConvexHull', 'FilledImage', 'ConvexImage');

figure; imshow(grayImg); hold on;
plot(mserCandidates, 'showPixelList', true, 'showEllipses', false);
title('MSER Regions'); hold off;

%% Stage 3: Geometric filtering
allBBoxes = vertcat(mserStats.BoundingBox);
w = allBBoxes(:,3);
h = allBBoxes(:,4);
ar_w_h = w ./ h;
ar_h_w = h ./ w;
areas = w .* h;

rejectIdx = (ar_w_h > 10) | (ar_h_w > 2) | (areas > 1000) ...
          | ([mserStats.Solidity] < 0.3) ...
          | ([mserStats.EulerNumber] < -4);

mserStats(rejectIdx) = [];
mserCandidates(rejectIdx) = [];

figure; imshow(grayImg); hold on;
plot(mserCandidates, 'showPixelList', true, 'showEllipses', false);
title('After Geometric Filtering'); hold off;

%% Stage 4: Character bounding boxes (no expansion)
charBoxesInitial = vertcat(mserStats.BoundingBox);
xmin = charBoxesInitial(:,1);
ymin = charBoxesInitial(:,2);
xmax = xmin + charBoxesInitial(:,3) - 1;
ymax = ymin + charBoxesInitial(:,4) - 1;

xmin = max(xmin, 1);
ymin = max(ymin, 1);
xmax = min(xmax, size(grayImg,2));
ymax = min(ymax, size(grayImg,1));

charBoxesInitial = [xmin ymin xmax - xmin + 1 ymax - ymin + 1];
imgCharBoxes = insertShape(origImg, 'Rectangle', charBoxesInitial, ...
    'Color', 'yellow', 'LineWidth', 3);

figure; imshow(imgCharBoxes);
title('Character Boxes (No Expansion)');

%% Stage 5: Additional false positive filtering
keepIdx = false(size(charBoxesInitial,1),1);
for i = 1:size(charBoxesInitial,1)
    wBox = charBoxesInitial(i,3);
    hBox = charBoxesInitial(i,4);
    if (hBox / wBox) < 3
        keepIdx(i) = true;
    end
end
charBoxesFiltered = charBoxesInitial(keepIdx,:);
imgCharFiltered = insertShape(origImg, 'Rectangle', charBoxesFiltered, ...
    'LineWidth', 3);
figure; imshow(imgCharFiltered);
title('Characters After False Positive Filtering');

%% Stage 6: Area filtering for small char blocks
areaKeepIdx = (charBoxesFiltered(:,3) .* charBoxesFiltered(:,4)) < 1000;
charBoxesFiltered = charBoxesFiltered(areaKeepIdx,:);
imgCharAreaFiltered = insertShape(origImg, 'Rectangle', charBoxesFiltered, ...
    'LineWidth', 3);
figure; imshow(imgCharAreaFiltered);
title('Characters After Area Filtering');

%% Stage 7: Overlap graph
overlapMat = bboxOverlapRatio(charBoxesFiltered, charBoxesFiltered);
n = size(overlapMat,1);
overlapMat(1:n+1:end) = 0;
charGraph = graph(overlapMat);
figure; plot(charGraph);
title('Graph of Connected Components');

%% Stage 8: Merge connected components
compIdx = conncomp(charGraph);
xminM = accumarray(compIdx', xmin, [], @min);
yminM = accumarray(compIdx', ymin, [], @min);
xmaxM = accumarray(compIdx', xmax, [], @max);
ymaxM = accumarray(compIdx', ymax, [], @max);
mergedTextBoxes = [xminM yminM xmaxM - xminM + 1 ymaxM - yminM + 1];

imgMerged = insertShape(origImg, 'Rectangle', mergedTextBoxes, ...
    'Color', 'yellow', 'LineWidth', 3);
figure; imshow(imgMerged);
title('Merged Text Boxes');

%% Stage 9: Expand merged boxes
expandFactor = 0.008;
xminE = max((1 - expandFactor) * xminM, 1);
yminE = max((1 - expandFactor) * yminM, 1);
xmaxE = min((1 + expandFactor) * xmaxM, size(grayImg,2));
ymaxE = min((1 + expandFactor) * ymaxM, size(grayImg,1));
expandedTextBoxes = [xminE yminE xmaxE - xminE + 1 ymaxE - yminE + 1];

imgExpanded = insertShape(origImg, 'Rectangle', expandedTextBoxes, ...
    'LineWidth', 3);
figure; imshow(imgExpanded);
title('Expanded Text Boxes');

%% Stage 10: OCR classification (requires pretrained models)
load lukasN.mat;
wordBoxes = [];
nonTextBoxes = [];
fidText = fopen('OCRWords_Pred_Text.txt','wt');
fidNonText = fopen('OCRWordsPred_NonText.txt','wt');

for k = 1:size(expandedTextBoxes,1)
    currBox = expandedTextBoxes(k,:);
    croppedRegion = imcrop(origImg, currBox);
    inputRegion = imresize(croppedRegion, [227 227]);
    
    feats = activations(convnet_nontextcl, inputRegion, featureLayer_nontextcl);
    [predLabel, ~, score] = predict(classifier_nontextcl, feats);
    
    if strcmp(char(predLabel), 'texts')
        wordBoxes = [wordBoxes; currBox];
        ocrRes = ocr(origImg, currBox);
        conf = mean(ocrRes.WordConfidences, 'omitnan');
        fprintf(fidText, '%d Word_Confidence=%.2f ', size(wordBoxes,1), conf);
        fprintf(fidText, '%s ', ocrRes.Words{:});
        fprintf(fidText, '\n\n');
    else
        nonTextBoxes = [nonTextBoxes; currBox];
        ocrRes = ocr(origImg, currBox);
        conf = mean(ocrRes.WordConfidences, 'omitnan');
        fprintf(fidNonText, '%d Word_Confidence=%.2f ', size(nonTextBoxes,1), conf);
        fprintf(fidNonText, '%s ', ocrRes.Words{:});
        fprintf(fidNonText, '\n\n');
    end
end
fclose(fidText);
fclose(fidNonText);

%% Stage 11: OCR confidence visualisation
ocrResults = ocr(origImg, expandedTextBoxes);
confScores = zeros(size(expandedTextBoxes,1),1);
for i = 1:numel(ocrResults)
    confScores(i) = mean(ocrResults(i).WordConfidences, 'omitnan');
end
imgOCR = insertObjectAnnotation(origImg, 'rectangle', ...
    expandedTextBoxes, confScores, 'FontSize', 55);
figure; imshow(imgOCR); title('OCR Results');

%% Stage 12: Top 30% OCR scores
[sortedConf, sortedIdx] = sort(confScores, 'descend');
topIdx = sortedIdx(1:floor(0.3 * numel(sortedIdx)));
imgOCRTop = insertObjectAnnotation(origImg, 'rectangle', ...
    expandedTextBoxes(topIdx,:), sortedConf(1:numel(topIdx)), ...
    'FontSize', 45);
figure; imshow(imgOCRTop);
title('Top 30% OCR Confidence');

%% Stage 13: Graph on detected word boxes
overlapWords = bboxOverlapRatio(wordBoxes, wordBoxes);
nW = size(overlapWords,1);
overlapWords(1:nW+1:end) = 0;
wordGraph = graph(overlapWords);
figure; plot(wordGraph);
title('Graph: Connected Components of Words');

compIdxW = conncomp(wordGraph);
xminW = accumarray(compIdxW', wordBoxes(:,1), [], @min);
yminW = accumarray(compIdxW', wordBoxes(:,2), [], @min);
xmaxW = accumarray(compIdxW', wordBoxes(:,1) + wordBoxes(:,3) - 1, [], @max);
ymaxW = accumarray(compIdxW', wordBoxes(:,2) + wordBoxes(:,4) - 1, [], @max);
finalWordBoxes = [xminW yminW xmaxW - xminW + 1 ymaxW - yminW + 1];

imgWords = insertShape(origImg, 'Rectangle', finalWordBoxes, 'LineWidth', 3);
figure; imshow(imgWords); title('Final Detected Text');
