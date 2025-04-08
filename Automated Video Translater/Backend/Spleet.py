import os
import logging
from spleeter.separator import Separator

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    separator = Separator('spleeter:2stems')
    input_audio = 'Backend/out.mp3'
    output_dir = 'Backend/output'

    os.makedirs(output_dir, exist_ok=True)
    separator.separate_to_file(input_audio, output_dir)
    print(f"Separated files are saved in: {output_dir}")
