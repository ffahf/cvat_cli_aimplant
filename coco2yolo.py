import json
import os

def coco_to_yolo_keypoints(coco_json_path, output_folder):
    # Load COCO JSON
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Create mapping from image ID to filename and image size
    image_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
    image_id_to_size = {img['id']: (img['width'], img['height']) for img in coco_data['images']}
    
    # Process annotations
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        keypoints = ann['keypoints']  # [x1, y1, v1, x2, y2, v2, ...]
        category_id = ann['category_id'] - 1  # YOLO classes start from 0
        width, height = image_id_to_size[image_id]
        
        # Convert bbox to YOLO format: [x_center, y_center, width, height] (normalized)
        x_min, y_min, box_w, box_h = ann['bbox']
        x_center = (x_min + box_w / 2) / width
        y_center = (y_min + box_h / 2) / height
        norm_box_w = box_w / width
        norm_box_h = box_h / height

        # Normalize keypoints
        normalized_keypoints = []
        for i in range(0, len(keypoints), 3):
            x, y = keypoints[i], keypoints[i+1]
            x /= width
            y /= height
            normalized_keypoints.extend([x, y])
        
        # Prepare YOLO format line
        yolo_line = f"{category_id} {x_center} {y_center} {norm_box_w} {norm_box_h} " + " ".join(map(str, normalized_keypoints)) + "\n"
        
        # Write to file
        yolo_filename = os.path.splitext(image_id_to_filename[image_id])[0] + '.txt'
        yolo_filepath = os.path.join(output_folder, yolo_filename)
        
        with open(yolo_filepath, 'a') as yolo_file:
            yolo_file.write(yolo_line)
    
    print(f"Conversion completed. YOLO keypoints saved in {output_folder}")

# Example usage
coco_json_path = 'pilot_cvat/label.json'
output_folder = 'yolo_keypoints5'
coco_to_yolo_keypoints(coco_json_path, output_folder)
