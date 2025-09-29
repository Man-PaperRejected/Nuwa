import json
import os
import shutil
from pathlib import Path

def copy_images_from_json(json_file_path: str, source_root_dir: str, destination_dir: str):
    """
    Reads image paths from a JSON file and copies the corresponding images to a destination directory.

    Args:
        json_file_path (str): The path to the JSON file.
        source_root_dir (str): The root directory where the images are stored.
                                 The 'image' field in JSON is relative to this directory.
        destination_dir (str): The directory where the images will be copied to.
    """
    json_file = Path(json_file_path)
    source_root = Path(source_root_dir)
    dest_dir = Path(destination_dir)

    if not json_file.is_file():
        print(f"Error: JSON file not found at {json_file}")
        return

    if not source_root.is_dir():
        print(f"Error: Source root directory not found at {source_root}")
        return

    # Create destination directory if it doesn't exist
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Ensured destination directory exists: {dest_dir}")

    copied_count = 0
    skipped_count = 0
    error_count = 0

    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"Read {len(data)} entries from {json_file}")

        for entry in data:
            if 'image' in entry:
                relative_image_path = entry['image']
                # Construct the full source path
                source_image_path = source_root / relative_image_path

                if source_image_path.is_file():
                    # Define the destination path. We can just copy it into the destination dir.
                    # If you want to preserve the directory structure, more logic is needed.
                    # For simplicity, we'll copy all images to the root of destination_dir
                    # and potentially rename if there are name conflicts (though unlikely with unique image names).
                    
                    # Extract just the filename to avoid potential path traversal issues
                    image_filename = source_image_path.name
                    destination_image_path = dest_dir / image_filename

                    # Avoid copying if the file already exists and is identical (optional, but good for repeated runs)
                    if destination_image_path.exists():
                        # You could add a check here to compare file sizes or checksums if needed
                        # print(f"Skipping, destination already exists: {destination_image_path}")
                        skipped_count += 1
                        continue

                    try:
                        shutil.copy2(source_image_path, destination_image_path)
                        # print(f"Copied: {source_image_path} -> {destination_image_path}")
                        copied_count += 1
                    except Exception as e:
                        print(f"Error copying {source_image_path} to {destination_image_path}: {e}")
                        error_count += 1
                else:
                    print(f"Warning: Image file not found at expected path: {source_image_path}")
                    error_count += 1
            else:
                print(f"Warning: 'image' field missing in an entry.")

    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_file}. Please check the file format.")
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return

    print("\n--- Copying Complete ---")
    print(f"Total entries processed: {len(data)}")
    print(f"Images successfully copied: {copied_count}")
    print(f"Images skipped (already exist): {skipped_count}")
    print(f"Errors encountered (file not found or copy failed): {error_count}")

# --- 使用示例 ---
if __name__ == "__main__":
    # 1. 指定你的JSON文件路径
    # 假设你的 JSON 文件名为 'dataset.json' 并且在当前脚本所在目录下
    json_file_to_process = '/data/code/VisionZip/coco_data/refcoco.json' 
   
    json_file_to_process =  '/data/code/VisionZip/coco_data/refcoco.json' 
    source_root_dir = '/data/dataset/coco2014' # 假设图片在 ./data/images/ 目录下
    destination_dir = './train2014/' 

    copy_images_from_json(json_file_to_process, source_root_dir, destination_dir)
