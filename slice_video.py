import argparse
import os
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input-video', required=True)
parser.add_argument('-o', '--output-dir', required=True)
parser.add_argument('-f', '--fps', required=True)
parser.add_argument('-s', '--size', required=True)
args = parser.parse_args()

base = os.path.basename(args.input_video).replace('.mp4', '')
output_path = f"{args.output_dir}/{base}_{args.fps}/images"
os.makedirs(output_path, exist_ok=True)

subprocess.run([
    'ffmpeg', '-i', args.input_video, '-vf', f'fps={args.fps}',
    '-q:v', '2', '-s', args.size,
    f"{output_path}/frame_%04d.jpg"
])