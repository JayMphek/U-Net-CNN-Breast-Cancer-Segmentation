# AI-Driven Mammography Analysis

## 📌 Project Overview
This research project focuses on developing an efficient, scalable AI diagnostic tool for mammography analysis. It is specifically designed to overcome critical constraints in the South African public sector, such as the shortage of radiologists and delayed diagnoses.

## ⚙️ System Architecture
The core architecture consists of a computationally efficient, two-stage AI pipeline:
* **Stage 1 (Segmentation):** U-Net architecture.
* **Stage 2 (Classification):** Random Forest classifier.

## 📊 Current Performance & Clinical Evaluation
While the initial study successfully demonstrated the technical feasibility of this two-stage approach, evaluation against clinical safety standards revealed significant performance failures that currently pose an unacceptable safety risk:
* **Segmentation Accuracy:** The U-Net model achieved a critically low Dice Score of 32%.
* **Classification Recall:** The system yielded a 50% Malignant Recall, meaning it currently misses half of all cancer cases.

## 🚀 Future Roadmap
To close the current safety gap and ensure a clinically safe detection rate of **>90%**, future work will pivot to advanced deep learning methodologies. The immediate priority is replacing the current classification model with a **dual-phased Convolutional Neural Network (CNN)**.
