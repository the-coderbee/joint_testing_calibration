import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm


def calculate_defect_area(mask_path):
    """Calculate defect area percentage from segmentation mask"""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    total_pixels = mask.size
    defect_pixels = np.count_nonzero(mask)
    return (defect_pixels / total_pixels) * 100


def create_regression_dataset(root_dir):
    """Create regression dataset from segmentation data"""
    data = []

    for split in ["train", "valid", "test"]:
        image_dir = os.path.join(root_dir, split, "images")
        mask_dir = os.path.join(root_dir, split, "labels")

        for img_file in tqdm(os.listdir(image_dir)):
            if img_file.endswith((".jpg", ".png")):
                img_path = os.path.join(image_dir, img_file)
                mask_path = os.path.join(mask_dir, img_file)

                if os.path.exists(mask_path):
                    defect_percent = calculate_defect_area(mask_path)
                    data.append(
                        {"image_path": img_path, "defect_percent": defect_percent}
                    )

    df = pd.DataFrame(data)
    df.to_csv("data/processed/regression_data.csv", index=False)
    return df


if __name__ == "__main__":
    create_regression_dataset("data/raw/weld_quality_inspection_segmentation")
