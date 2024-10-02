import os
import csv

# Read from .log file
def parse_log_file(log_file_path):
    with open(log_file_path, 'r') as file:
        lines = file.readlines()
        
    miou = float(lines[-1].split()[-1].strip('%')) / 100.0
    return miou

def generate_csv(compressor, error_bound_list, base_dir, current_folder):
    # CSV file name and path
    csv_file_name = f"{compressor}_results.csv"
    csv_file_path = os.path.join(base_dir, csv_file_name)
    
    # CSV Header
    header = ["Model", "Compressor", "error_bound", "epochs", "miou", "run"]
    
    # Open and Write CSV
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        
        # Iterate over each error bound in folder
        for run, error_bound in enumerate(error_bound_list, start=1):
            folder_name = f"{compressor}_{error_bound}"
            log_file_path = os.path.join(base_dir, current_folder, folder_name, "logs", "valid.log")

            # mIoU from log file - parse class
            miou = parse_log_file(log_file_path)

            # Write row for current error bound
            row = [
                "EfficientVit", compressor, error_bound, 420, miou, run
            ]
            writer.writerow(row)
        
    print(f"CSV file created at: {csv_file_path}")
    
    
# Define your compressors and their respective error bounds
compressors = {
    "zfp": ["1E1", "1E2", "1E3", "1E4", "1E5", "1E6", "1E7"],
    "sz3": ["1E1", "1E2", "1E3", "1E4", "1E5", "1E6", "1E7"],  
    "jpeg": ["Q0", "Q10", "Q20", "Q30", "Q40", "Q50", "Q60", "Q70", "Q80", "Q90", "Q100"]
}


# Compressor folder directories
base_directory = "/home/marque6/MLBD/LossyUGVPaper/EfficientVitCat/runtest"

folders = {
    "zfp": "zfp",
    "sz3": "sz3",
    "jpeg": "jpeg"
}

for compressor, error_bounds in compressors.items():
    generate_csv(compressor, error_bounds, base_directory, folders[compressor])
