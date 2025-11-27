import os
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm

def augment_and_save_images_with_masks(input_image_folder, input_mask_folder, output_image_folder, output_mask_folder, total_samples=300):
   
    transform = A.Compose([  
        A.HorizontalFlip(p=0.3),
        A.Rotate(limit=10, p=0.3),])

    # Get list of image and mask files
    image_files = sorted([f for f in os.listdir(input_image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    mask_files = sorted([f for f in os.listdir(input_mask_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    
    if len(image_files) != len(mask_files):
        print("Warning: Mismatch in the number of images and masks!")

    
    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_mask_folder, exist_ok=True)

    num_original_images = len(image_files)

    
    num_augs_per_image = max(1, total_samples // num_original_images)

    index = 0
    for img_name, mask_name in tqdm(zip(image_files, mask_files), total=num_original_images):
        image_file = os.path.join(input_image_folder, img_name)
        mask_file = os.path.join(input_mask_folder, mask_name)

        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)  # Load mask in original color (not grayscale)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)  
        if image is None or mask is None:
            print(f"Skipping: {img_name} (image/mask not found or invalid)")
            continue
        #image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
        #mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)  
        for _ in range(num_augs_per_image):
            augmented = transform(image=image, mask=mask)
            augmented_image = augmented['image']
            augmented_mask = augmented['mask']

         
            if augmented_image is None or augmented_mask is None:
                continue

            
            output_image_file = os.path.join(output_image_folder, f"augmented_{index}.jpg")
            output_mask_file = os.path.join(output_mask_folder, f"augmented_{index}.png")

            cv2.imwrite(output_image_file, cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))   
            cv2.imwrite(output_mask_file, cv2.cvtColor(augmented_mask, cv2.COLOR_RGB2BGR))  

            index += 1
            if index >= total_samples:
                break  # Stop when we reach the desired number of samples

        if index >= total_samples:
            break  # Stop if we have reached 2000 samples

if __name__ == "__main__":
    # Define paths
    input_image_folder = "E:/Final_Annotation_intersection/orginal_images/train/"
    input_mask_folder = "E:/Final_Annotation_intersection/Binary/trainset/"
    output_image_folder = "E:/aug_images/"
    output_mask_folder = "E:/aug_masks/"

    total_samples = 300

    augment_and_save_images_with_masks(input_image_folder, input_mask_folder, output_image_folder, output_mask_folder, total_samples)