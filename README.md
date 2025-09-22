# 🎯 Road Pothole Detection 
A test of using the yolo11m model to detect road potholes in 100 epochs.  
Model: yolo11m.pt  
Dataset: pothole-detection URL:https://www.kaggle.com/datasets/andrewmvd/pothole-detection
# 📸 Examples
Here are some example outputs from the detector:
![val_batch2_pred](https://github.com/user-attachments/assets/42483840-62ce-46de-9200-9dc9b15bbc9e)
# 📊 Training Metrics
<img width="2400" height="1200" alt="results" src="https://github.com/user-attachments/assets/260052f4-a4f9-426d-a943-9ede06c43d3f" />
<img width="500" height="400" alt="BoxP_curve" src="https://github.com/user-attachments/assets/ae57e095-92cf-483d-98a4-9aafb577ed46" />
<img width="500" height="400" alt="BoxR_curve" src="https://github.com/user-attachments/assets/bc8531f7-4575-436c-bfcd-c4677bc78e20" />
<img width="500" height="400" alt="BoxPR_curve" src="https://github.com/user-attachments/assets/797b73fd-17fa-4cbc-bd9c-80085d24b9cd" />
<img width="500" height="400" alt="BoxF1_curve" src="https://github.com/user-attachments/assets/b2c32869-61ea-4a5d-99bf-7cefb2b246f2" />

### The training curves indicate:

📉 Losses (box, cls, DFL) consistently decrease across training and validation.  
✅ Precision and recall stabilize around ~0.70 after 80 epochs, but exhibit a fluctuation in the later stages of training.  
📈 The model achieved a final precision of 0.743, recall of 0.732, mAP50 of 0.806 and mAP50-95 of 0.537，indicating the model has acquired a preliminary  capability for pothole identification.Despite the model's unstable performance due to the limited training epochs, there is still significant room for improvement in pothole recognition. However, as a first attempt, the results are acceptable.
