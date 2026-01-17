import os
from pathlib import Path
import math

import csv

import cv2
from tqdm.auto import tqdm

from ultralytics import YOLO

from ytsv.slide_utils import extract_pages_and_audios
from ytsv.system_utils import detect_systems_by_batch, crop_systems, detect_staff_heights_by_batch, resize_systems
from ytsv.utils import load_yolo_models


def process_single_video(
  metadata:list[str], 
  dataset_dir:Path, 
  yolo_models:list[YOLO], 
  target_height:int=18, 
  device:str='cpu'
):
  yolo_system, yolo_staff_height = yolo_models

  yt_id, *_, staff_count = metadata

  staff_count_dir = dataset_dir / staff_count
  staff_count_dir.mkdir(exist_ok=True)

  mp4_dir = staff_count_dir / 'mp4'
  mp4_dir.mkdir(exist_ok=True)

  seg_dir = staff_count_dir / 'segments' / yt_id
  seg_dir.mkdir(parents=True, exist_ok=True)

  mp4_path = mp4_dir / f'{yt_id}.mp4'
  
  default_mp4_path = dataset_dir / 'mp4' / f'{yt_id}.mp4'
  if default_mp4_path.exists():
    mp4_path = default_mp4_path.rename(mp4_path)
  
  extract_pages_and_audios( mp4_path, seg_dir )

  page_images = sorted( (seg_dir / 'images' / 'original').glob('*.png') )
  yolo_bbox_paths = detect_systems_by_batch(page_images, yolo_system, device=device)
  
  crop_image_paths = []
  for yolo_bbox_path in yolo_bbox_paths:
    crop_image_paths += crop_systems(yolo_bbox_path)

  average_staff_heights = detect_staff_heights_by_batch(crop_image_paths, yolo_staff_height, device=device)
  resize_systems(crop_image_paths, average_staff_heights, target_height=target_height)


def process_videos_from_scratch(
  dataset_dir:Path, 
  metaddata_path:Path, 
  checkpoint_dir:Path, 
  target_height:int=18, 
  device:str='cpu'
):
  """
  :param dataset_dir: path to the dataset directory
  :param metaddata_path: path to the metadata file
  :param checkpoint_dir: path to the YOLO model weights for staff height detection
  """

  with open(metaddata_path, 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    metadata = list(reader)
  
  yolo_models = load_yolo_models(checkpoint_dir)

  for row in tqdm(metadata):
    process_single_video(
      row, 
      dataset_dir, 
      yolo_models,
      target_height=target_height,
      device=device
    )