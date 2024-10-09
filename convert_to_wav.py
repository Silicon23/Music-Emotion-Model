"""
Convert files to WAV format using ffmpeg and resamples to the sampling rate in byol-a/config.yaml.

Usage:
    ```python -u convert_to_wav.py /path/to/input/directory /path/to/output/directory```
"""
import os
import ffmpeg
import yaml
import argparse
from tqdm import tqdm

def convert_to_wav(input_dir, output_dir):
    # Load the sample rate from BYOL-A YAML configuration file
    with open('./byol-a/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    sample_rate = config.get('sample_rate', 16000)  # Default to 16000 if not specified

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get list of mp3 files
    files = [f for f in os.listdir(input_dir) if f.endswith('.mp3') or f.endswith('.mp4') or f.endswith('.wav')]

    # Iterate over all files in the input directory with a progress bar
    for filename in tqdm(files, desc="Converting to WAV"):
        input_path = os.path.join(input_dir, filename)
        output_filename = os.path.splitext(filename)[0] + '.wav'
        output_path = os.path.join(output_dir, output_filename)

        # Convert to wav using ffmpeg
        (
            ffmpeg
            .input(input_path)
            .output(output_path, ar=sample_rate, ac=1)
            .run(quiet=True, overwrite_output=True)
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert MP3 files to WAV format.')
    parser.add_argument('input_dir', type=str, help='Directory containing MP3 files')
    parser.add_argument('output_dir', type=str, help='Directory to save converted WAV files')
    args = parser.parse_args()

    convert_to_wav(args.input_dir, args.output_dir)
