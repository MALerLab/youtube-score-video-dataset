from datetime import datetime
from pathlib import Path
import logging

import json


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