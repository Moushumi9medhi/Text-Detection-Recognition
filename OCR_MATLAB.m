% Load an image
close all;

I = imread('Image1.png');
results = ocr(I);

word = results.Words{1}

wordBBox = results.WordBoundingBoxes(2,:)

figure;
Iname = insertObjectAnnotation(I, 'rectangle', wordBBox, word);
imshow(Iname);

lowConfidenceIdx = results.CharacterConfidences < 0.5;

lowConfBBoxes = results.CharacterBoundingBoxes(lowConfidenceIdx, :);

lowConfVal = results.CharacterConfidences(lowConfidenceIdx);

str      = sprintf('confidence = %f', lowConfVal);
Ilowconf = insertObjectAnnotation(I, 'rectangle', lowConfBBoxes, str);

figure;
imshow(Ilowconf);
I = imread('image2.jpg');
I = rgb2gray(I);

figure;
imshow(I)
results = ocr(I);

results.Text

results = ocr(I, 'TextLayout', 'Block');

results.Text
th = graythresh(I);
BW = im2bw(I, th);

figure;
imshowpair(I, BW, 'montage');
Icorrected = imtophat(I, strel('disk', 15));

th  = graythresh(Icorrected);
BW1 = im2bw(Icorrected, th);

figure;
imshowpair(Icorrected, BW1, 'montage');
marker = imerode(Icorrected, strel('line',10,0));
Iclean = imreconstruct(marker, Icorrected);

th  = graythresh(Iclean);
BW2 = im2bw(Iclean, th);

figure;
imshowpair(Iclean, BW2, 'montage');
results = ocr(BW2, 'TextLayout', 'Block');

results.Text
regularExpr = '\d';

bboxes = locateText(results, regularExpr, 'UseRegexp', true);

digits = regexp(results.Text, regularExpr, 'match');

Idigits = insertObjectAnnotation(I, 'rectangle', bboxes, digits);

figure;
imshow(Idigits);
results = ocr(BW2, 'CharacterSet', '0123456789', 'TextLayout','Block');

results.Text
[sortedConf, sortedIndex] = sort(results.CharacterConfidences, 'descend');

indexesNaNsRemoved = sortedIndex( ~isnan(sortedConf) );

topTenIndexes = indexesNaNsRemoved(1:10);

digits = num2cell(results.Text(topTenIndexes));
bboxes = results.CharacterBoundingBoxes(topTenIndexes, :);

Idigits = insertObjectAnnotation(I, 'rectangle', bboxes, digits);

figure;
imshow(Idigits);
blobAnalyzer = vision.BlobAnalysis('MaximumCount', 500);

[area, centroids, roi] = step(blobAnalyzer, BW1);

img = insertShape(I, 'rectangle', roi);
figure;
imshow(img);
areaConstraint = area > 300;

roi = double(roi(areaConstraint, :));
 
img = insertShape(I, 'rectangle', roi);
figure;
imshow(img); 
width  = roi(:,3);
height = roi(:,4);
aspectRatio = width ./ height;
 
roi = roi( aspectRatio > 0.25 & aspectRatio < 1 ,:);
 
img = insertShape(I, 'rectangle', roi);
figure;
imshow(img);
text = deblank( {results.Text} );
img  = insertObjectAnnotation(I, 'rectangle', roi, text);

figure;
imshow(img)
