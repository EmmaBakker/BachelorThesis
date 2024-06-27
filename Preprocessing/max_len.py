import numpy as np
import json

def calculate_sequence_lengths(data):
    """
    Calculate the lengths of sequences in the dataset.
    """
    lengths = [len(sequence) for sequence in data]
    return lengths

def analyze_lengths(lengths, title):
    """
    Print statistics and log the distribution of sequence lengths.
    """
    lengths = np.array(lengths)
    max_length = np.max(lengths)
    average_length = np.mean(lengths)
    median_length = np.median(lengths)
    percentile_90 = np.percentile(lengths, 90)  # 90th percentile length
    
    print(f"Maximum sequence length for {title}: {max_length}")
    print(f"Average sequence length for {title}: {average_length:.2f}")
    print(f"Median sequence length for {title}: {median_length}")
    print(f"90th percentile sequence length for {title}: {percentile_90}")
    
    # Log detailed statistics
    print(f"Lengths less than 10: {np.sum(lengths < 10)}")
    print(f"Lengths between 10 and 20: {np.sum((lengths >= 10) & (lengths < 20))}")
    print(f"Lengths between 20 and 50: {np.sum((lengths >= 20) & (lengths < 50))}")
    print(f"Lengths between 50 and 100: {np.sum((lengths >= 50) & (lengths < 100))}")
    print(f"Lengths between 100 and 200: {np.sum((lengths >= 100) & (lengths < 200))}")

def main(filenames):
    """
    Main function to process the data.
    """
    for filename, title in filenames.items():
        print(f"\nAnalyzing {title}...")
        # Open the JSON file and load data
        with open(filename, 'r') as file:
            data = json.load(file)
        
        lengths = calculate_sequence_lengths(data)
        analyze_lengths(lengths, title)

if __name__ == "__main__":
    # Dictionary of file paths and their respective titles
    filenames = {
        '/Users/emmabakker/Desktop/Studie/Jaar3/Thesis/Transfomer/en_train_data.json': 'English Train/Test Data',
        '/Users/emmabakker/Desktop/Studie/Jaar3/Thesis/Transfomer/hi_train_data.json': 'Hindi Data'
    }
    main(filenames)
