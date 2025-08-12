#define _CRT_SECURE_NO_WARNINGS
#define _MAIN

#include <opencv/highgui.h>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include "region.h"
#include "mser.h"
#include "max_meaningful_clustering.h"
#include "region_classifier.h"
#include "group_classifier.h"
#include "utils.h"

using namespace std;
using namespace cv;

#define NUM_FEATS 11
#define THRESH_EA 0.5
#define THRESH_SF 0.999999999

// ---------------------------
// Load image and preprocess
// ---------------------------
void preprocessImage(const string& path, Mat& grayImg, Mat& labImg, Mat_<double>& gradMag) {
    Mat img = imread(path);
    cvtColor(img, grayImg, CV_BGR2GRAY);
    cvtColor(img, labImg, CV_BGR2Lab);
    get_gradient_magnitude(grayImg, gradMag);
}

// ---------------------------
// MSER extraction + filtering
// ---------------------------
void runMSER(const Mat& grayImg, const Mat& labImg, const Mat_<double>& gradMag,
             vector<Region>& outRegions, double& maxStroke) {

    MSER mserDetector(false, 25, 0.000008, 0.03, 1, 0.7);
    mserDetector((uchar*)grayImg.data, grayImg.cols, grayImg.rows, outRegions);

    for (auto& r : outRegions) r.er_fill(grayImg);

    maxStroke = 0;
    for (int i = (int)outRegions.size() - 1; i >= 0; --i) {
        outRegions[i].extract_features(labImg, grayImg, gradMag);
        if ((outRegions[i].stroke_std_ / outRegions[i].stroke_mean_ > 0.8) ||
            (outRegions[i].num_holes_ > 2) ||
            (outRegions[i].bbox_.width <= 3) || (outRegions[i].bbox_.height <= 3))
            outRegions.erase(outRegions.begin() + i);
        else
            maxStroke = max(maxStroke, outRegions[i].stroke_mean_);
    }
}

// ---------------------------
// Multi-feature clustering
// ---------------------------
void performClustering(vector<Region>& regions, double maxStroke,
                       GroupClassifier& groupCls, Mat& allSegs, vector<vector<int>>& finalGroups) {

    MaxMeaningfulClustering mmc(METHOD_METR_SINGLE, METRIC_SEUCLIDEAN);
    Mat coMatrix = Mat::zeros((int)regions.size(), (int)regions.size(), CV_64F);

    int dims[NUM_FEATS] = {3,3,3,3,3,3,3,3,3,5,5};

    for (int feat = 0; feat < NUM_FEATS; feat++) {
        unsigned int N = regions.size();
        if (N < 3) break;

        int dim = dims[feat];
        t_float* data = (t_float*)malloc(dim * N * sizeof(t_float));
        int idx = 0;

        for (int r = 0; r < regions.size(); r++) {
            data[idx]   = (regions[r].bbox_.x + regions[r].bbox_.width / 2.0) / allSegs.cols;
            data[idx+1] = (regions[r].bbox_.y + regions[r].bbox_.height / 2.0) / allSegs.rows;
            switch (feat) {
                case 0:  data[idx+2] = regions[r].intensity_mean_ / 255.0; break;
                case 1:  data[idx+2] = regions[r].boundary_intensity_mean_ / 255.0; break;
                case 2:  data[idx+2] = regions[r].bbox_.y / (double)allSegs.rows; break;
                case 3:  data[idx+2] = (regions[r].bbox_.y + regions[r].bbox_.height) / (double)allSegs.rows; break;
                case 4:  data[idx+2] = max(regions[r].bbox_.height, regions[r].bbox_.width) / (double)max(allSegs.rows, allSegs.cols); break;
                case 5:  data[idx+2] = regions[r].stroke_mean_ / maxStroke; break;
                case 6:  data[idx+2] = regions[r].area_ / (double)(allSegs.rows * allSegs.cols); break;
                case 7:  data[idx+2] = (regions[r].bbox_.height * regions[r].bbox_.width) / (double)(allSegs.rows * allSegs.cols); break;
                case 8:  data[idx+2] = regions[r].gradient_mean_ / 255.0; break;
                case 9:
                    data[idx+2] = regions[r].color_mean_[0] / 255.0;
                    data[idx+3] = regions[r].color_mean_[1] / 255.0;
                    data[idx+4] = regions[r].color_mean_[2] / 255.0;
                    break;
                case 10:
                    data[idx+2] = regions[r].boundary_color_mean_[0] / 255.0;
                    data[idx+3] = regions[r].boundary_color_mean_[1] / 255.0;
                    data[idx+4] = regions[r].boundary_color_mean_[2] / 255.0;
                    break;
            }
            idx += dim;
        }

        vector<vector<int>> clusters;
        mmc(data, N, dim, METHOD_METR_SINGLE, METRIC_SEUCLIDEAN, &clusters);

        for (auto& c : clusters) {
            accumulate_evidence(&c, 1, &coMatrix);
            if (groupCls(&c, &regions) >= THRESH_SF) finalGroups.push_back(c);
        }

        Mat tmpSeg = Mat::zeros(allSegs.size(), CV_8UC3);
        drawClusters(tmpSeg, &regions, &clusters);
        tmpSeg.copyTo(allSegs(Rect(320*feat, 0, 320, 240)));

        free(data);
    }
}

// ---------------------------
// Character + word detection
// ---------------------------
void detectAndSave(const vector<Region>& regions, const string& outFile) {
    vector<Rect> charBoxes;
    for (const auto& r : regions)
        charBoxes.emplace_back(r.bbox_.x, r.bbox_.y, r.bbox_.width, r.bbox_.height);

    ofstream fout(outFile);
    for (const auto& box : charBoxes)
        fout << box.x << " " << box.y << " " << box.width << " " << box.height << endl;
}

int main(int argc, char** argv) {
    Mat grayImg, labImg;
    Mat_<double> gradMag;
    preprocessImage("InputImg.png", grayImg, labImg, gradMag);

    RegionClassifier regCls("boost_train/trained_boost_char.xml", 0);
    GroupClassifier grpCls("boost_train/trained_boost_groups.xml", &regCls);

    vector<Region> regions;
    double maxStroke = 0;
    runMSER(grayImg, labImg, gradMag, regions, maxStroke);

    Mat allSegs = Mat::zeros(240, 320*NUM_FEATS, CV_8UC3);
    vector<vector<int>> finalGroups;
    performClustering(regions, maxStroke, grpCls, allSegs, finalGroups);

    Mat segOutput = Mat::zeros(grayImg.size(), CV_8UC3);
    drawClusters(segOutput, &regions, &finalGroups);
    imwrite("segmentation_output.png", segOutput);

    detectAndSave(regions, "character_boxes.txt");
    waitKey(0);
    return 0;
}
