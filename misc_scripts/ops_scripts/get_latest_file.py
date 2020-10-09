import sys
import os
import re
import shutil

def get_latest_file(dir_path:str) -> str:
    """Considers all filenames containing numbers, and returns filename with larget numnber"""
    sample_filenames = [filename for filename in os.listdir(dir_path) if re.search('\d+', filename)]
    sample_filenames.sort(key=lambda filename: int(re.search('\d+', filename)[0]))
    latest_filename = sample_filenames[-1]
    return latest_filename

if __name__ == '__main__':
    dir_path = sys.argv[1]
    latest_filename = get_latest_file(dir_path)
    print(os.path.join(dir_path, latest_filename))
    # duplicate this file as 'final'
    ext = os.path.splitext(latest_filename)
    shutil.copy(os.path.join(dir_path, latest_filename), os.path.join(dir_path, f'final_{latest_filename}'))