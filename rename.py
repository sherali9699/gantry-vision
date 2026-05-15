import os
import shutil

def rename_and_copy_images(src_dir, dest_dir):
    # Create destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print(f"Created directory: {dest_dir}")

    # List all files and filter for .png
    files = [f for f in os.listdir(src_dir) if f.lower().endswith('.png')]
    files.sort()  # Sort to ensure consistent ordering

    if not files:
        print("No .png files found in the source directory.")
        return

    print(f"Found {len(files)} images. Starting process...")

    for index, filename in enumerate(files, start=1):
        # Generate the new name (e.g., image01.png, image02.png)
        # using :02d for 2-digit padding (01, 02...)
        new_name = f"image{index:03d}.png"
        
        src_path = os.path.join(src_dir, filename)
        dest_path = os.path.join(dest_dir, new_name)

        # Use shutil.copy2 to preserve metadata while copying
        shutil.copy2(src_path, dest_path)
        
    print(f"Success! Images moved and renamed in {dest_dir}")

# Define paths
source_folder = './yolo-train-images-with-version-06/'
destination_folder = './yolo-train-images-06-rename/'

if __name__ == "__main__":
    rename_and_copy_images(source_folder, destination_folder)