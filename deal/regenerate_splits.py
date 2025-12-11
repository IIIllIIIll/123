import csv
import random
from pathlib import Path

# Define paths
script_dir = Path(__file__).parent
input_path = script_dir.parent.parent / 'output' / 'labels.csv'
output_path = script_dir.parent.parent / 'output' / 'labels_pro.csv'

def main():
    random.seed(88)
    # Read the existing CSV
    with open(input_path, 'r', newline='') as infile:
        reader = csv.reader(infile)
        header = next(reader)  # Skip header
        data = list(reader)
    
    # Separate by label
    label_0 = [row for row in data if row[1] == '0']
    label_1 = [row for row in data if row[1] == '1']
    
    # Function to split group
    def split_group(group):
        random.shuffle(group)
        total = len(group)
        train_size = int(total * 0.7)
        val_size = int(total * 0.1)
        test_size = total - train_size - val_size
        
        for i in range(train_size):
            group[i][2] = 'train'
        for i in range(train_size, train_size + val_size):
            group[i][2] = 'val'
        for i in range(train_size + val_size, total):
            group[i][2] = 'test'
        return group
    
    # Split each group
    split_0 = split_group(label_0)
    split_1 = split_group(label_1)
    
    # Combine
    new_data = split_0 + split_1
    
    # Write to new CSV
    with open(output_path, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)
        writer.writerows(new_data)
    
    print(f"New labels file created at {output_path}")

if __name__ == "__main__":
    main()