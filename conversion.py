import os
import shutil
import pandas as pd
import pydicom
import cv2
from tqdm import tqdm

DATASET_PATH = "CBIS-DDSM"
CSV_DIR = "csv"
OUTPUT_DIR = "dataset"

def create_folders(split):
    split_dir = os.path.join(OUTPUT_DIR, split)
    if os.path.exists(split_dir):
        shutil.rmtree(split_dir)
        
    for label in ["benign", "malignant"]:
        os.makedirs(os.path.join(OUTPUT_DIR, split, label), exist_ok=True)

def convert_dcm(dcm_path, save_path):
    try:
        ds = pydicom.dcmread(dcm_path)
        img = ds.pixel_array

        img = (img - img.min()) / (img.max() - img.min()) * 255
        img = img.astype("uint8")

        cv2.imwrite(save_path, img)
        return True
    except Exception as e:
        return False

def process_csv(csv_file, split):
    if not os.path.exists(csv_file):
        return

    df = pd.read_csv(csv_file)

    for _, row in tqdm(df.iterrows(), total=len(df), desc=os.path.basename(csv_file)):
        pathology = row["pathology"]

        # Map BENIGN_WITHOUT_CALLBACK to benign
        if pathology in ["BENIGN", "BENIGN_WITHOUT_CALLBACK"]:
            label = "benign"
        elif pathology == "MALIGNANT":
            label = "malignant"
        else:
            continue

        full_path = row["cropped image file path"]
        if pd.isna(full_path):
            continue

        try:
            folder_name = str(full_path).split("/")[0]
        except:
            continue

        folder_path = os.path.join(DATASET_PATH, folder_name)

        if not os.path.exists(folder_path):
            continue

        found_crop = False
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".dcm"):
                    dcm_path = os.path.join(root, file)
                    try:
                        # Quickly inspect the header to verify this is the "cropped images" series
                        ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
                        desc = getattr(ds, "SeriesDescription", "")
                        
                        if desc == "cropped images":
                            save_name = f"{folder_name}.png"
                            save_path = os.path.join(OUTPUT_DIR, split, label, save_name)
                            if convert_dcm(dcm_path, save_path):
                                found_crop = True
                                break # Found the crop
                    except:
                        pass
            if found_crop:
                break

def main():
    create_folders("train")
    create_folders("test")

    csvs = [
        ("mass_case_description_train_set.csv", "train"),
        ("calc_case_description_train_set.csv", "train"),
        ("mass_case_description_test_set.csv", "test"),
        ("calc_case_description_test_set.csv", "test")
    ]

    for filename, split in csvs:
        process_csv(os.path.join(CSV_DIR, filename), split)
        
    print("Dataset generation completed!")

if __name__ == "__main__":
    main()