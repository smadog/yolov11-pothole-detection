from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'



def plot_detection_results(image, results):
    """使用 matplotlib 显示检测结果"""
    # 转换 BGR 到 RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 创建图表
    plt.figure(figsize=(12, 8))

    # 处理每个检测结果
    for result in results:
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)

            # 显示原始图像
            plt.imshow(image_rgb)

            # 绘制边界框和标签
            for i, (box, conf, cls_id) in enumerate(zip(xyxy, confs, cls_ids)):
                x1, y1, x2, y2 = map(int, box)
                class_name = result.names[cls_id]
                confidence = float(conf)

                # 绘制矩形框
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     fill=False, edgecolor='red', linewidth=2)
                plt.gca().add_patch(rect)

                # 添加标签
                plt.text(x1, y1 - 10, f'{class_name}: {confidence:.2f}',
                         bbox=dict(facecolor='red', alpha=0.5),
                         fontsize=10, color='white')

            plt.axis('off')
            plt.title(f'检测结果 - 共检测到 {len(boxes)} 个目标', fontsize=14)

    return plt


# 加载模型和图像
model = YOLO(
    r"D:\xly\project\pycharm project\yolov11\ultralytics-main\runs\detect\pothole_detection_v2\weights\best.pt")
image_path = r"D:\xly\project\pycharm project\yolov11\ultralytics-main\yolo_formatted_dataset\val\images\potholes45.png"

# 进行预测
results = model(image_path)
image = cv2.imread(image_path)

# 显示结果
plt = plot_detection_results(image, results)
plt.tight_layout()
plt.show()
