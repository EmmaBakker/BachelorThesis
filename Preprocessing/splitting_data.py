import json
import argparse
import random
import os

def split_data(input_data, output_dir, language, seed, train_ratio=0.8, validation_ratio=0.1):
    # Initialize seed for reproducibility
    random.seed(seed)

    # Load the data
    with open(input_data, 'r') as f:
        sentences = json.load(f)

    # Shuffle the data
    random.shuffle(sentences)

    # Calculate split points
    total_sentences = len(sentences)
    split_point_1 = int(total_sentences * train_ratio)
    split_point_2 = int(total_sentences * (train_ratio + validation_ratio))

    # Split the data
    train_set = sentences[:split_point_1]
    validation_set = sentences[split_point_1:split_point_2]
    test_set = sentences[split_point_2:]

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)

    # Define file paths
    output_train = os.path.join(output_dir, f"{language}_train_set.json")
    output_validation = os.path.join(output_dir, f"{language}_val_set.json")
    output_test = os.path.join(output_dir, f"{language}_test_set.json")

    # Save the splits to files
    with open(output_train, 'w', encoding='utf-8') as f:
        json.dump(train_set, f)
    print(f"Train set saved to {output_train}")
    
    with open(output_validation, 'w', encoding='utf-8') as f:
        json.dump(validation_set, f)
    print(f"Validation set saved to {output_validation}")
    
    with open(output_test, 'w', encoding='utf-8') as f:
        json.dump(test_set, f)
    print(f"Test set saved to {output_test}")

    print(f"Data successfully split and saved to {output_train}, {output_validation}, and {output_test}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, required=True, help="File with sentences to split.")
    parser.add_argument("--output_dir", type=str, default=os.getcwd(), help="Directory to save the output files.")
    parser.add_argument("--language", type=str, required=True, help="Language prefix for the output files.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()
    split_data(args.input_data, args.output_dir, args.language, args.seed)
