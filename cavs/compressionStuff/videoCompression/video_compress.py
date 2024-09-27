import cv2
import os
import subprocess
import argparse
from tqdm import tqdm

def compress_frames(input_dir, output_file, frame_rate=30, crf=23):
    if not os.path.isdir(input_dir):
        raise ValueError(f"Input directory '{input_dir}' does not exist.")

    image_files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    if not image_files:
        raise ValueError(f"No image files found in {input_dir}")

    first_image = cv2.imread(os.path.join(input_dir, image_files[0]))
    height, width = first_image.shape[:2]

    ffmpeg_cmd = [
        'ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}',
        '-pix_fmt', 'bgr24',
        '-r', str(frame_rate),
        '-i', '-',
        '-c:v', 'libopenh264',
        '-crf', str(crf),
        '-preset', 'medium',
        '-pix_fmt', 'yuv420p',
        output_file
    ]

    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    for image_file in tqdm(image_files, desc="Processing frames"):
        img = cv2.imread(os.path.join(input_dir, image_file))
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        process.stdin.write(img.tobytes())

    process.stdin.close()
    stderr = process.communicate()[1]
    if process.returncode != 0:
        print(f"FFmpeg error: {stderr.decode()}")
    else:
        print(f"Compression completed. Output saved as {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Compress image frames to H.264 video using OpenCV and FFmpeg.")
    parser.add_argument("input_dir", help="Directory containing input frames")
    parser.add_argument("output_file", help="Output video file name")
    parser.add_argument("--frame_rate", type=int, default=30, help="Frame rate of the output video")
    parser.add_argument("--crf", type=int, default=23, help="CRF value (0-51, lower is higher quality)")

    args = parser.parse_args()

    compress_frames(args.input_dir, args.output_file, args.frame_rate, args.crf)

if __name__ == "__main__":
    main()