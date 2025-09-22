from ultralytics import YOLO
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def main():
    # 加载模型（从上次的最佳权重继续训练）
    model = YOLO(r"D:\xly\project\pycharm project\yolov11\ultralytics-main\runs\detect\pothole_detection2\weights\best.pt")

    # 训练配置
    results = model.train(
        # 基础配置
        data=r"D:\xly\project\pycharm project\yolov11\ultralytics-main\yolo_formatted_dataset\data.yaml",
        epochs=100,
        imgsz=640,
        batch=4,
        workers=0,

        # 数据增强配置
        augment=True,  # 启用数据增强
        scale=0.5,  # 缩放增强
        fliplr=0.5,  # 水平翻转

        # 优化器配置
        optimizer='auto',  # 自动选择优化器
        lr0=0.01,  # 初始学习率
        lrf=0.01,  # 最终学习率
        momentum=0.937,  # 动量
        weight_decay=0.0005,  # 权重衰减
        warmup_epochs=3.0,  # 热身epoch
        warmup_momentum=0.8,  # 热身动量
        warmup_bias_lr=0.1,  # 热身偏置学习率

        # 训练策略
        patience=50,  # 早停耐心
        dropout=0.2,  # Dropout
        erasing=0.4,  # 随机擦除
        auto_augment='randaugment',  # 自动增强

        # 其他配置
        name='pothole_detection_v2',
        exist_ok=True,  # 覆盖现有运行
        resume=False,  # 不继续训练，重新开始
        verbose=True  # 详细输出
    )


if __name__ == '__main__':
    main()