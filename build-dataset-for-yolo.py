import os
import shutil
import random
from pathlib import Path

# --- CONFIGURATION ---
image_folder = 'yolo-train-images'
label_folder = 'gantry-annotation-labels'
image_exts = ['.jpg', '.jpeg', '.png']
train_split = 0.8  
dataset_name = "gantry_upload"

def build_yolo_dataset():
    # 1. Setup Folders
    base_path = Path(dataset_name)
    for split in ['train', 'val']:
        (base_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (base_path / split / 'labels').mkdir(parents=True, exist_ok=True)

    # 2. Identify image-label pairs from your specific directories
    all_images = [f for f in os.listdir(image_folder) if Path(f).suffix.lower() in image_exts]
    
    valid_pairs = []
    for img_name in all_images:
        # Match the filename stem (e.g., 'frame1') with a .txt in the label folder
        label_name = Path(img_name).stem + ".txt"
        label_path = os.path.join(label_folder, label_name)
        
        if os.path.exists(label_path):
            valid_pairs.append((img_name, label_name))
    
    if not valid_pairs:
        print(f"❌ No matching pairs found! Check if filenames in '{image_folder}' match '{label_folder}'")
        return

    # 3. Shuffle and Split
    random.seed(42) # For reproducible splits
    random.shuffle(valid_pairs)
    split_point = int(len(valid_pairs) * train_split)
    
    train_set = valid_pairs[:split_point]
    val_set = valid_pairs[split_point:]

    # 4. Copy Files to the new structure
    def copy_to_structure(data_set, folder_name):
        for img, lbl in data_set:
            # Source paths
            src_img = os.path.join(image_folder, img)
            src_lbl = os.path.join(label_folder, lbl)
            
            # Destination paths
            shutil.copy(src_img, base_path / folder_name / 'images' / img)
            shutil.copy(src_lbl, base_path / folder_name / 'labels' / lbl)

    copy_to_structure(train_set, 'train')
    copy_to_structure(val_set, 'val')

    # 5. Create data.yaml (formatted for your Kaggle environment)
    yaml_content = f"""
path: /kaggle/input/{dataset_name}
train: train/images
val: val/images

nc: 2
names: ['microwave', 'package_box']
"""
    with open(base_path / 'data.yaml', 'w') as f:
        f.write(yaml_content.strip())

    print(f"✅ Success! Created {dataset_name} in your directory.")
    print(f"📈 Total Pairs: {len(valid_pairs)} | Train: {len(train_set)} | Val: {len(val_set)}")

if __name__ == "__main__":
    build_yolo_dataset()