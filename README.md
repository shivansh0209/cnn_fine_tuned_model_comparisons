# Image Classification Using Transfer Learning: VGG16 vs MobileNetV2

## 📌 Project Overview

This project focuses on improving image classification performance by comparing a **baseline CNN model** with **fine-tuned pretrained deep learning models**. The main objective was to analyze the impact of **data augmentation** and **transfer learning** on model accuracy and interpretability.

The project evaluates and compares:

- A baseline CNN model
- Fine-tuned **VGG16**
- Fine-tuned **MobileNetV2**

Additionally, **Grad-CAM visualization** was used to understand where the models focus while making predictions.

---

# 🎯 Objectives

The key objectives of this project were:

- Build a baseline image classification model.
- Analyze the impact of data augmentation on model performance.
- Implement transfer learning using pretrained CNN architectures.
- Compare the performance of VGG16 and MobileNetV2.
- Visualize model decision-making using Grad-CAM.
- Identify which model provides better feature extraction and classification ability.

---

# 🏗️ Project Workflow
Dataset
|
|
Data Preprocessing
|
|
Data Augmentation
|
|
+----------------------+
| |
Baseline CNN Transfer Learning
| |
| +-------+-------+
| | |
| VGG16 MobileNetV2
| | |
+--------------+---------------+
|
|
Performance Comparison
|
|
Grad-CAM Analysis


---

# 📊 Model Experiments

## 1. Baseline Model

A custom CNN architecture was first trained as a reference model.

### Without Data Augmentation

- Maximum Accuracy: **~65%**

This established the initial performance of the model without any additional optimization techniques.

---

## 2. Baseline Model + Data Augmentation

Data augmentation techniques were introduced to improve generalization.

Techniques applied:

- Image rotation
- Image flipping
- Image transformation
- Other augmentation operations

### Result

- Accuracy improved from **~65% → ~75%**

This demonstrated that data augmentation helped the model learn more robust features and reduced overfitting.

---

# 🔥 Transfer Learning Experiments

## 3. Fine-Tuned VGG16

The pretrained VGG16 architecture was fine-tuned on the dataset.

Approach:

- Used pretrained ImageNet weights
- Applied data augmentation
- Fine-tuned model layers for task-specific feature learning

### Result

Maximum Accuracy:
87%


VGG16 showed a significant improvement compared to the baseline model due to its ability to extract powerful visual features.

---

## 4. Fine-Tuned MobileNetV2

MobileNetV2 was then implemented and fine-tuned using the same training strategy.

Advantages of MobileNetV2:

- Lightweight architecture
- Uses inverted residual blocks
- Uses depthwise separable convolutions
- Efficient feature extraction

### Result

Maximum Accuracy:
90%+


MobileNetV2 achieved the best classification performance among all tested models.

---

# 📈 Performance Comparison

| Model | Data Augmentation | Accuracy |
|---|---|---|
| Baseline CNN | ❌ No | ~65% |
| Baseline CNN | ✅ Yes | ~75% |
| Fine-Tuned VGG16 | ✅ Yes | ~87% |
| Fine-Tuned MobileNetV2 | ✅ Yes | 90%+ |

---

# 🔍 Model Interpretability Using Grad-CAM

To understand model predictions, Grad-CAM visualizations were generated for both fine-tuned models.

## Observations

### VGG16

- Focused on broader image regions.
- Sometimes attended to less relevant areas.
- Feature localization was comparatively weaker.

### MobileNetV2

- Focused more accurately on important image regions.
- Highlighted relevant features required for classification.
- Showed better attention localization.

This helped explain why MobileNetV2 achieved slightly higher accuracy compared to VGG16.

---

# 🧠 Key Learnings

Through this project, the following concepts were explored:

- CNN model development
- Data augmentation
- Transfer learning
- Fine-tuning pretrained architectures
- VGG16 architecture
- MobileNetV2 architecture
- Model comparison
- Grad-CAM visualization
- Model interpretability

---

# 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- OpenCV

---

# 🚀 Future Improvements

Possible improvements:

- Experiment with other pretrained architectures:
  - EfficientNet
  - ResNet
  - ConvNeXt

- Perform hyperparameter optimization.
- Apply advanced augmentation techniques.
- Deploy the trained model using a web application.

---

# 📌 Conclusion

This project demonstrated that combining **data augmentation** with **transfer learning** significantly improves image classification performance.

While VGG16 provided strong classification capability, MobileNetV2 achieved superior accuracy with efficient feature extraction. Grad-CAM analysis further showed that MobileNetV2 focused more effectively on relevant image regions, making it the best-performing model in this comparison.