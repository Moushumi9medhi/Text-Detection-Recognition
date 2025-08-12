# 🔍 Text Detection in GUI Applications 

<div align="center">

![Text Detection Banner](https://img.shields.io/badge/🔍Text%20Detection%20%26%20Recognition-Computer%20Vision%20%7C%20Machine%20Learning%20%7C%20Non--ML-blue?style=for-the-badge&logo=opencv&logoColor=white)
![Industry Transfer](https://img.shields.io/badge/🏭%20Industry%20Transfer-Altair%20Engineering-orange?style=for-the-badge&logo=altair&logoColor=white)

*Advanced text detection, matching, and recognition system for GUI application screen images*

[📝 Documentation](Documentation.pdf)

</div>

---

## 🎯 Overview

This project presents a comprehensive solution for **text detection, text matching, and image matching** within GUI application screen images. This work explores both traditional computer vision techniques and deep learning approaches.

### 🤝 Industry Collaboration
**Sponsoring Agency & Technology Transfer**: Altair Engineering India Pvt. Ltd.

---

## 📑 Table of Contents
- [🎯 Overview](#-overview)
- [🤝 Industry Collaboration](#-industry-collaboration)
- [📋 Prerequisites](#-prerequisites)
- [🛠 Technologies Used](#-technologies-used)
- [🔧 Core Algorithms Implemented](#-core-algorithms-implemented)
- [🧩 Methodology](#-methodology)
  - [🖼️ Non-ML Modules](#️-non-ml-modules)
    - [MSER (Maximally Stable Extremal Regions)](#1️⃣-mser-maximally-stable-extremal-regions)
    - [SWT (Stroke Width Transform)](#2️⃣-swt-stroke-width-transform)
    - [Morphological Transformations & Edge Proposals](#3️⃣-morphological-transformations--edge-proposals)
    - [Optical Character Recognition (OCR)](#4️⃣-optical-character-recognition-ocr)
  - [🤖 ML Modules](#-ml-modules)
    - [Traditional ML](#-traditional-ml)
    - [Deep Learning Hybrid](#-deep-learning-hybrid)
- [🔍 Comparative State-of-the-Art Methods](#-comparative-state-of-the-art-methods)
- [👥 Contributors](#-contributors)
- [🤝 Contributing](#-contributing)
- [📬 Contact](#-contact)

---

## 🛠 Technologies Used
- **Programming Languages**: Python, MATLAB, C++
- **Libraries**:
  - OpenCV
  - NumPy
  - Boost
  - LIBSVM
- **Tools**: MATLAB MEX, ASP .NET

---

## 📋 Prerequisites

```bash
# Python dependencies
pip install opencv-python numpy matplotlib

# MATLAB (optional)
# OpenCV C++ libraries
# Boost libraries for C++ implementation
```

---

## 🔧 Core Algorithms Implemented

| Algorithm / Technique | Implementation | 
|-----------------------|----------------|
| **MSER** (Maximally Stable Extremal Regions) | OpenCV, MATLAB, Boost | 
| **SWT** (Stroke Width Transform) | OpenCV + Boost, MATLAB + C++ MEX, Python + OpenCV| 
| **Morphological Transformations** | OpenCV, Python | 
| **Canny Edge Detection & Proposal Filtering** | OpenCV | 
| **OCR** (Optical Character Recognition) | MATLAB, ASP.NET | 
| **Histogram Comparison** | MATLAB, Python | 
| **Color Correlogram** | MATLAB | 
| **Local Binary Patterns (LBP)** | MATLAB, Python | 
| **ADABOOST** | MATLAB | 
| **SVM** | MATLAB, OpenCV | 
| **LIBSVM** | C++, Python | 
| **CNN-SVM** | Python |

---

## 🧩 Methodology

We integrate both **non-machine-learning** and **machine-learning** modules for robust text detection, recognition, and classification.

### 🖼️ Non-ML Modules

#### 1️⃣ MSER (Maximally Stable Extremal Regions)
- **Strengths**: Reliable for consistent color text in still frames.
- **Limitations**: Slower for large, high-resolution images.

#### 2️⃣ SWT (Stroke Width Transform)
- **Strengths**: Effective for uniform stroke-width text.
- **Challenges**: High-dimensional images could cause infinite loops, requiring cropping.

#### 3️⃣ Morphological Transformations & Edge Proposals
- **Operations**: Top-hat filtering, erosion, dilation, Gaussian smoothing, thresholding, Canny edge detection.

#### 4️⃣ Optical Character Recognition (OCR)
- **Performance**: Better results on cropped patches; full images produced more false positives.

---

### 🤖 ML Modules

#### 🔹 Traditional ML


- **Algorithms**:  
  - **ADABOOST**: Ensemble of weak classifiers; less effective for text detection.  
  - **SVM**: Used handcrafted features (histogram, color correlogram, LBP); ~66% loss rate.  
  - **LIBSVM**: RBF kernel version improved classification.

```
🎯 Features Tested:
├── Color Correlogram
├── Histogram Features
├── Local Binary Patterns (LBP)
└── Morphological Features
```

```
📚 Dataset: Chars74K
├── 64 classes 
├── 7,705 natural images
├── 3,410 hand-drawn characters
└── 62,992 synthesized characters
```
- Chars74K dataset: [Download here](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/).
  - Preprocessing: 95% of images were tightly fitted into rectangular contours, with 5% excluded due to poor outline fitting.

#### 🔹 Deep Learning Hybrid
- **CNN-SVM**: CNN feature extractor + SVM classifier.  
  - Dataset: 406 images (203 text, 203 icons).  
  - Achieved higher accuracy than standalone SVM or LIBSVM.


---

## 🔍 Comparative State-of-the-Art Methods

### 🔹 [Lukas Neumann Text Detector](https://ieeexplore.ieee.org/document/7333861)
- **Category**: Scene Text Detection (State-of-the-Art)  
- **Key Idea**: Uses a segmentation-based approach with a combination of connected component analysis and text grouping for robust detection in natural scenes.  
- **Strengths**: 
  - The method detects initial text hypotheses in a single pass using a character stroke area estimation feature and region-based approach.
- **Limitations**:  
  - Available implementation relies on pre-trained models without detailed training dataset documentation.  
  - Detection algorithm fails for the full image-requires cropped patches from the original image.
- **Implementation**:
  - OpenCV + C++
- **Use in This Work**:  
  - **Reviewed for comparison purposes only**.  
  - Performance insights informed the evaluation criteria and benchmarking strategy for the custom methods developed in this project.

---

## 👥 Contributors

- **Moushumi Medhi**  
- **Dr. Rajiv Ranjan Sahay**

---

## 🤝 Contributing

We welcome contributions! 

1. 🍴 Fork the repository
2. 🌟 Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. 💾 Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. 📤 Push to the branch (`git push origin feature/AmazingFeature`)
5. 🔄 Open a Pull Request

---

## 📬 Contact

<div align="center">

[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:medhi.moushumi@iitkgp.ac.in)

</div>

---

<div align="center">
<sub>Built with ❤️ at IIT Kharagpur · Department of Electrical Engineering | 2017 DIMG Project · Computational Vision Lab</sub>

</div>
