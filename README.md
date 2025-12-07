# 📌 Pneumonia Detection from Chest X-Ray Images

**Deep Learning | CNN | Transfer Learning (DenseNet121, MobileNetV2, VGG16)**

A deep-learning-based medical imaging project to classify chest X-rays into Pneumonia and Normal using state-of-the-art CNN architectures. This repository contains reproducible training notebooks with preprocessing, fine-tuning, evaluation, and saved model files.


* * *

## 📂 Dataset

The dataset used is the **Kaggle Pneumonia Chest X-Ray Dataset**:

| Resource | Link |
|---------|------|
| Chest X-Ray Pneumonia Dataset | [Kaggle Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) |


It contains three folders:

`/train /test /val`

Because the provided validation set had **only 16 images**, this project **recreated a larger validation set by 20% data(validation_split=0.2) from the training set**, ensuring stable training and reliable evaluation.

* * *

## 🚀 Model

### **1\. DenseNet121 (Main Model)(Pneumonia_DenseNet121_model.ipynb)**

-   Pretrained on ImageNet
    
-   Frozen base → trained classifier head
    
-   Unfrozen final layers → fine-tuned
    
-   Strong regularization & data augmentation
    
-   Achieved high validation performance
    
    

* * *

## 🧠 Full Pipeline Features

✔ Load & preprocess dataset  
✔ Custom **train/validation split**  
✔ Data augmentation  
✔ Transfer learning  
✔ Fine-tuning  
✔ Callbacks  
– EarlyStopping  
– ModelCheckpoint  
✔ Training curves visualization  
✔ Classification report & confusion matrix  
✔ Model saving  
✔ Compatibility with GitHub / Google Colab / VSCode

* * *


## 📊 Training Summary (Updated Validation Set)

After increasing the validation set size:

-   Validation accuracy became **more stable**
    
-   Loss stopped fluctuating heavily
    
-   Generalization improved (less overfitting)
    
-   Evaluation metrics became **more trustworthy**
    

This fixed the original issue where the validation set had only **16 images**, which was too small for reliable metrics.

* * *

## 🔧 Technologies Used


| Component | Library |
| --- | --- |
| **Deep Learning** | TensorFlow / Keras |
| **Image Loading & Augmentation** | Keras `ImageDataGenerator` |
| **Numerical Computing** | NumPy |
| **Model Evaluation** | scikit-learn (confusion matrix, classification report) |
| **Visualization** | Matplotlib |
| **Dataset Access** | Kaggle API |
| **File & Directory Operations** | os, shutil |
| **Execution Environment** | Google Colab |

* * *

## ▶️ How to Pull Into VSCode

`git clone https://github.com/Leyan365/ML-PNEUMONIA-Detection.git`

Then open the folder in VSCode.

* * *

## 

.

* * *

## 📌 Results (Actual Metrics)

### **✔ Test Set Performance**

## 

| Metric | Value |
| --- | --- |
| **Accuracy** | **90.54%** |
| **AUC** | **0.9652** |
| **Loss** | 0.4625 |


### **✔ Classification Report**

## 

              `precision    recall  f1-score   support  NORMAL          0.97       0.77      0.86       234 PNEUMONIA       0.88       0.98      0.93       390  accuracy                            0.91       624 macro avg       0.92       0.88      0.89       624 weighted avg    0.91       0.91      0.90       624`



### **✔ Confusion Matrix**


`[[181  53]  [  6 384]]`

-   **NORMAL:** 181 correct, 53 misclassified
    
-   **PNEUMONIA:** 384 correct, 6 misclassified
    


### **✔ Interpretation**


-   Very **high sensitivity (recall)** for Pneumonia → important for medical diagnosis
    
-   Normal cases had lower recall due to some being predicted as pneumonia
    
-   Overall strong generalization and high reliability

* * *


## 🏁 Conclusion

This project demonstrates:

-   How to properly create a deep learning pipeline
    
-   How to fix dataset issues (like too-small validation sets)
    
-   How to use advanced transfer learning (DenseNet121)
    
-   How to train, fine-tune, evaluate, and deploy medical models
    

The DenseNet121 model provides strong diagnostic performance and generalizes well thanks to proper validation and augmentation.

