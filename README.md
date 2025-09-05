
# Automatic Detection of Hard Exudates in Retinal Images

This project is based on my Bachelor’s Thesis, focusing on the **automatic detection of hard exudates** in retinal fundus images. Exudates are an important biomarker for the early diagnosis of **Diabetic Retinopathy (DR)**.  

The pipeline was fully developed in **Python** and integrates **classical image processing**, **segmentation (FCM)**, and **machine learning (SVM)**, together with a **Streamlit interface** for visualization.

---

## 📌 Features
- Preprocessing of retinal fundus images (normalization, CLAHE, morphology).
- Optic disc detection (classical method + **U-Net**).
- Blood vessel segmentation (Frangi filter).
- Candidate region extraction using **Fuzzy C-Means (FCM)**.
- Feature extraction (color, texture, contrast-based descriptors).
- Classification using **Support Vector Machine (SVM)** with RBF kernel.
- Performance evaluation with ROC, Precision-Recall, and confusion matrix.
- Streamlit-based interface with two modes:
  - **Screening** (maximize recall).
  - **Monitoring** (maximize precision).

---

