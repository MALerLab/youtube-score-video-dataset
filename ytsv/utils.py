import os
import sys
from io import StringIO
import contextlib
import logging

from datetime import datetime
from pathlib import Path

import json
import requests

from ultralytics import YOLO


dprint = lambda x: print(json.dumps(x, indent=2))
get_ts = lambda: datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

filter_files = lambda path_gen: sorted(p for p in path_gen if p.is_file())
filter_dirs = lambda path_gen: sorted(p for p in path_gen if p.is_dir())


def get_logger(file_path:Path) -> logging.Logger:
  logger = logging.getLogger(__name__)
  logging.basicConfig(
    filename=str(file_path), 
    format='[%(asctime)s][%(levelname)s] >> %(msg)s', 
    level=logging.INFO
  )
  
  return logger


def format_logger_msg(logger_prefix, msg_dict):
  msg_list = []
  
  logger_prefix = f'[{logger_prefix}]'
  msg_list.append(logger_prefix)
  
  msg_list += list( msg_dict.values() )
  
  return ': '.join(msg_list)


YOLO_MODELS_URLS = {
  'ls-yolo-system-v2.0.0.pt': 'https://github.com/MALerLab/ls-yolo/releases/download/system-v2/ls-yolo-system-v2.0.0.pt',
  'ls-yolo-staff-height-v2.0.0.pt': 'https://github.com/MALerLab/ls-yolo/releases/download/staff-height-v2/ls-yolo-staff-height-v2.0.0.pt'
}


def download_yolo_model_checkpoint(model_name:str, checkpoint_path:Path):
  r = requests.get( YOLO_MODELS_URLS[model_name], allow_redirects=True )

  if r.status_code != 200:
    raise Exception(f'Failed to download YOLO model checkpoint: {model_name} from {YOLO_MODELS_URLS[model_name]}')
  
  checkpoint_path.parent.mkdir(exist_ok=True)
  with open(checkpoint_path, 'wb') as f:
    f.write(r.content)


def load_yolo_model(checkpoint_name, checkpoint_dir:Path) -> YOLO:
  checkpoint_path = checkpoint_dir / checkpoint_name

  if not checkpoint_path.exists():
    download_yolo_model_checkpoint(checkpoint_name, checkpoint_path)
  
  yolo_model = YOLO( checkpoint_path )

  return yolo_model


def load_yolo_models(checkpoint_dir:Path) -> list:
  yolo_system = load_yolo_model('ls-yolo-system-v2.0.0.pt', checkpoint_dir)
  yolo_staff_height = load_yolo_model('ls-yolo-staff-height-v2.0.0.pt', checkpoint_dir)

  return [ yolo_system, yolo_staff_height ]