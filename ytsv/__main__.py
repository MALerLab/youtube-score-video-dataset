from pathlib import Path
import argparse

from . import process_videos_from_scratch

def create_argparser():
  parser = argparse.ArgumentParser(
    prog='YouTube Score Video Dataset Preprocessor',
    description='MP4 to cropped score images, audio, and metadata',
  )
  
  parser.add_argument('-d', '--dataset-dir', type=str, required=True, help='Path to dataset directory')
  parser.add_argument('-m', '--metadata-path', type=str, required=True, help='Path to list of video metadata in .cmsv format')
  parser.add_argument('-c', '--checkpoint-dir', type=str, default='checkpoints', help='Path to YOLO model checkpoints directory')
  
  parser.add_argument('--target-height', type=int, default=18, help='Target staff height for resizing cropped images, default is 18')
  parser.add_argument('--device', type=str, required=False, default='cpu', help='Device to run YOLO models on, e.g., "cpu" or "cuda"')

  return parser


if __name__ == '__main__':
  parser = create_argparser()
  
  args = parser.parse_args()
  
  process_videos_from_scratch(
    dataset_dir=Path(args.dataset_dir),
    metaddata_path=Path(args.metadata_path),
    checkpoint_dir=Path.cwd() / 'checkpoints',
    target_height=args.target_height,
    device=args.device
  )