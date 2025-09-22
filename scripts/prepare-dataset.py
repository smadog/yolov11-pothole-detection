import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def setup_directories():
    """设置目录路径."""
    # 源数据集路径
    source_dir = Path("dataset")
    image_source_dir = source_dir / "images"
    annot_source_dir = source_dir / "annotations"

    # 输出路径
    output_dir = Path("../yolo_formatted_dataset")

    return source_dir, image_source_dir, annot_source_dir, output_dir


def get_class_names_from_annotations(annot_dir):
    """从标注文件中自动提取所有类别名称."""
    class_names = set()
    xml_files = list(annot_dir.glob("*.xml"))

    for xml_file in xml_files[:100]:  # 检查前100个文件以确定类别
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            for obj in root.findall("object"):
                class_name = obj.find("name").text
                class_names.add(class_name)

        except Exception as e:
            print(f"读取 {xml_file} 时出错: {e}")
            continue

    # 将集合转换为排序后的列表，确保顺序一致
    class_names = sorted(list(class_names))
    return class_names


def convert_voc_to_yolo(xml_path, class_name_to_id):
    """将单个VOC XML文件转换为YOLO格式."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 获取图片尺寸
        size = root.find("size")
        img_width = int(size.find("width").text)
        img_height = int(size.find("height").text)

        yolo_lines = []

        # 遍历所有object标签
        for obj in root.findall("object"):
            class_name = obj.find("name").text

            if class_name not in class_name_to_id:
                print(f"警告: 在文件 {xml_path.name} 中发现未知类别 '{class_name}'，已跳过")
                continue

            class_id = class_name_to_id[class_name]

            # 获取边界框坐标
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            # 计算YOLO格式的归一化坐标
            center_x = (xmin + xmax) / 2 / img_width
            center_y = (ymin + ymax) / 2 / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height

            # 确保坐标在[0,1]范围内
            center_x = max(0, min(1, center_x))
            center_y = max(0, min(1, center_y))
            width = max(0, min(1, width))
            height = max(0, min(1, height))

            yolo_lines.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")

        return yolo_lines

    except Exception as e:
        print(f"转换文件 {xml_path} 时出错: {e}")
        return []


def ensure_directory_exists(file_path):
    """确保文件所在的目录存在."""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def process_single_file(xml_file, image_source_dir, output_split_dir, class_name_to_id):
    """处理单个XML文件."""
    stem = xml_file.stem  # 获取文件名（不含扩展名）

    # 查找对应的图片文件
    img_file = None
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"]:
        potential_file = image_source_dir / (stem + ext)
        if potential_file.exists():
            img_file = potential_file
            break

    if not img_file:
        print(f"警告: 未找到 {stem} 的图片文件，跳过")
        return False

    # 转换XML到YOLO格式
    yolo_lines = convert_voc_to_yolo(xml_file, class_name_to_id)
    if not yolo_lines:
        print(f"警告: {stem} 没有有效的标注，跳过")
        return False

    # 创建输出目录（确保目录存在）
    images_output_dir = output_split_dir / "images"
    labels_output_dir = output_split_dir / "labels"
    images_output_dir.mkdir(parents=True, exist_ok=True)
    labels_output_dir.mkdir(parents=True, exist_ok=True)

    # 复制图片文件
    img_dest = images_output_dir / img_file.name
    try:
        shutil.copy2(img_file, img_dest)
    except Exception as e:
        print(f"复制图片 {img_file} 到 {img_dest} 时出错: {e}")
        return False

    # 保存YOLO格式标注
    label_dest = labels_output_dir / f"{stem}.txt"
    try:
        with open(label_dest, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))
    except Exception as e:
        print(f"写入标注文件 {label_dest} 时出错: {e}")
        return False

    return True


def create_data_yaml(output_dir, class_names):
    """创建YOLO数据集配置文件."""
    yaml_content = f"""# YOLO 数据集配置文件
path: {output_dir.absolute()}
train: train/images
val: val/images

# 类别数量
nc: {len(class_names)}

# 类别名称
names:
"""
    for idx, name in enumerate(class_names):
        yaml_content += f"  {idx}: {name}\n"

    yaml_file = output_dir / "data.yaml"

    # 确保目录存在
    yaml_file.parent.mkdir(parents=True, exist_ok=True)

    with open(yaml_file, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"配置文件已创建: {yaml_file}")


def main():
    """主函数."""
    print("开始准备YOLO格式数据集...")

    # 设置目录
    source_dir, image_source_dir, annot_source_dir, output_dir = setup_directories()

    # 检查源目录是否存在
    if not source_dir.exists():
        print(f"错误: 源数据集目录 {source_dir} 不存在!")
        return

    if not annot_source_dir.exists():
        print(f"错误: 标注目录 {annot_source_dir} 不存在!")
        return

    if not image_source_dir.exists():
        print(f"错误: 图片目录 {image_source_dir} 不存在!")
        return

    # 自动检测类别名称
    print("正在自动检测类别名称...")
    class_names = get_class_names_from_annotations(annot_source_dir)

    if not class_names:
        print("错误: 未检测到任何类别名称，请手动指定类别")
        # 如果自动检测失败，可以在这里手动指定
        class_names = ["crack", "pothole", "flooded", "blockage"]  # 根据你的实际情况修改

    class_name_to_id = {name: idx for idx, name in enumerate(class_names)}
    print(f"检测到的类别: {class_name_to_id}")

    # 创建输出目录结构（提前创建所有需要的目录）
    (output_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (output_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)

    # 获取所有XML文件
    xml_files = list(annot_source_dir.glob("*.xml"))
    print(f"找到 {len(xml_files)} 个XML标注文件")

    if not xml_files:
        print("错误: 未找到任何XML标注文件!")
        return

    # 随机打乱文件顺序
    random.seed(42)  # 设置随机种子以确保可重现
    random.shuffle(xml_files)

    # 划分训练集和验证集 (80% 训练, 20% 验证)
    split_idx = int(len(xml_files) * 0.8)
    train_files = xml_files[:split_idx]
    val_files = xml_files[split_idx:]

    print(f"数据集划分: {len(train_files)} 训练, {len(val_files)} 验证")

    # 处理训练集
    train_count = 0
    print("正在处理训练集...")
    for i, xml_file in enumerate(train_files):
        if process_single_file(xml_file, image_source_dir, output_dir / "train", class_name_to_id):
            train_count += 1
        if (i + 1) % 100 == 0:  # 每处理100个文件打印一次进度
            print(f"已处理 {i + 1}/{len(train_files)} 个训练文件")

    # 处理验证集
    val_count = 0
    print("正在处理验证集...")
    for i, xml_file in enumerate(val_files):
        if process_single_file(xml_file, image_source_dir, output_dir / "val", class_name_to_id):
            val_count += 1
        if (i + 1) % 100 == 0:  # 每处理100个文件打印一次进度
            print(f"已处理 {i + 1}/{len(val_files)} 个验证文件")

    # 创建data.yaml配置文件
    create_data_yaml(output_dir, class_names)

    print("\n处理完成!")
