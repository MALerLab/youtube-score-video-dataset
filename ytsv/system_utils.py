from pathlib import Path
import math

import cv2
from tqdm.auto import tqdm

from ultralytics import YOLO


def zip_bboxs_confs(output):
  bboxs = output.boxes.xyxy.int().tolist()
  confs = output.boxes.conf.tolist()

  bboxs = [ (*coords, conf) for coords, conf in zip(bboxs, confs) ]
  # sort by y, x
  bboxs = sorted( bboxs, key=lambda x: (x[1], x[0]) )

  return bboxs


def save_bboxs(bboxs, file_path):
  with open(file_path, 'w') as f:
    for bbox in bboxs:
      f.write(f'{" ".join(map(str, bbox))}\n')


def load_bboxs(file_path):
  bboxs = []

  with open(file_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
      bbox = [ p for p in line.strip().split(' ') ] # lx, ly, rx, ry, conf
      bbox = [ int(p) for p in bbox[:4] ] + [ float(bbox[4]) ]
      bboxs.append( bbox )
  
  return bboxs


def inference_by_batch(image_fns:list[Path], yolo:YOLO, batch_size:int=64, custom_process=None, device:str='cpu'):
  """
  :param image_fns: list of image file paths
  :param yolo: instance of YOLO model
  :param batch_size: batch size for YOLO inference, default is 64
  """
  num_batches = math.ceil(len(image_fns)/batch_size)

  outputs = []

  for cur_batch_idx in tqdm(range(num_batches), leave=False, desc="YOLO Inf"):
    batch = image_fns[cur_batch_idx*batch_size:(cur_batch_idx+1)*batch_size]
    
    if custom_process:
      batch_paths = batch
      batch = [ custom_process(b) for b in batch ]
    
    
    batch_output = yolo(batch, device=device, verbose=False)
    
    if custom_process:
      for bo, bp in zip(batch_output, batch_paths):
        bo.path = str(bp)

    outputs += batch_output
  
  return outputs


def process_yolo_system_output(output):
  bboxs = output.boxes.xyxy

  # skip if no bbox
  if len(bboxs) < 1:
    return False

  bboxs = zip_bboxs_confs(output)

  image_path = Path(output.path)
  yolo_bbox_path = image_path.with_name( image_path.stem + '_yolo_bboxs.txt' )

  save_bboxs(bboxs, yolo_bbox_path)

  return yolo_bbox_path


def detect_systems_by_batch(image_fns:list[Path], yolo:YOLO, batch_size:int=64, device:str='cpu'):
  """
  :param image_fns: list of image file paths
  :param yolo: instance of YOLO model
  :param batch_size: batch size for YOLO inference, default is 64
  """
  
  outputs = inference_by_batch(
    image_fns, 
    yolo=yolo, 
    batch_size=batch_size,
    device=device
  )

  # move model back to cpu
  if device != 'cpu':
    yolo.to('cpu')
  
  yolo_bbox_paths = []
  
  for output in outputs:
    yolo_bbox_path = process_yolo_system_output(output)
    if yolo_bbox_path:
      yolo_bbox_paths.append(yolo_bbox_path)
  

  return yolo_bbox_paths


def crop_systems(yolo_bbox_path:Path, ignore_existing:bool=False, conf_threshold:float=0.4):
  """
  :param yolo_bbox_path: path to the YOLO bbox annotation file
  :param ignore_existing: if True, skip cropping if cropped image already exists
  :param conf_threshold: confidence threshold for bbox selection
  """
  bboxs = load_bboxs(yolo_bbox_path) # [ (lx, ly, rx, ry, conf), ... ]

  image_path = yolo_bbox_path.name.replace('_yolo_bboxs.txt', '.png')
  image_path = yolo_bbox_path.with_name(image_path)
  image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

  crop_image_dir = image_path.parent.parent / 'cropped'
  crop_image_dir.mkdir(exist_ok=True, parents=False)

  crop_image_paths = []
  system_bboxs = []

  bboxs = [ b for b in bboxs if b[-1] >= conf_threshold ]

  for i, (lx, ly, rx, ry, conf) in enumerate(bboxs):
    # save cropped iamge
    crop_image_path = crop_image_dir / f'{image_path.stem}:{str(i).zfill(4)}.png'
    if ignore_existing and crop_image_path.exists():
      crop_image = cv2.imread(crop_image_path, cv2.IMREAD_UNCHANGED)
    
    else:
      crop_image = image[ly:ry, lx:rx]
      cv2.imwrite(crop_image_path, crop_image)
    
    crop_image_paths.append(crop_image_path)
    system_bboxs.append( (lx, ly, rx, ry, conf) )
  
  system_bboxs_path = yolo_bbox_path.name.replace('_yolo_bboxs.txt', '_system_bboxs.txt')
  system_bboxs_path = yolo_bbox_path.with_name(system_bboxs_path)
  save_bboxs(system_bboxs, system_bboxs_path)
  
  return crop_image_paths


def process_yolo_staff_height_output(output):
  bboxs = output.boxes.xyxy

  # skip if no bbox
  if len(bboxs) < 1:
    return False

  bboxs = zip_bboxs_confs(output)

  image_path = Path(output.path)
  staff_height_path = image_path.with_name( image_path.stem + '_staff_heights.txt' )

  save_bboxs(bboxs, staff_height_path)

  average_staff_height = sum([ y2 - y1 for (_, y1, _, y2, _) in bboxs ])
  average_staff_height /= len(bboxs)

  return average_staff_height


def detect_staff_heights_by_batch(crop_image_fns:list[Path], yolo:YOLO, batch_size:int=64, device:str='cpu'):
  """
  :param crop_image_fns: list of cropped image file paths
  :param yolo: instance of YOLO model
  :param batch_size: batch size for YOLO inference, default is 64
  """
  def crop_left_half(image_fn:Path):
    img = cv2.imread(image_fn, cv2.IMREAD_UNCHANGED)
    return img[:, :img.shape[1]//2]  # left half

  outputs = inference_by_batch(
    crop_image_fns, 
    yolo=yolo, 
    batch_size=batch_size, 
    custom_process=crop_left_half,
    device=device
  )
  
  staff_heights = []
  
  for output in outputs:
    average_staff_height = process_yolo_staff_height_output(output)
    staff_heights.append(average_staff_height)
  
  return staff_heights


def resize_systems(crop_image_fns:list[Path], staff_heights:list[float], target_height:int=18):
  """
  :param crop_image_fns: list of cropped image file paths
  :param staff_heights: list of average staff heights corresponding to the cropped images
  :param target_height: target height for resizing, default is 18
  """
  for crop_image_fn, staff_height in zip(crop_image_fns, staff_heights):
    if not staff_height:
      # need logging
      continue

    img = cv2.imread(crop_image_fn, cv2.IMREAD_UNCHANGED)
    ratio = target_height / staff_height
    r_w = int(img.shape[1] * ratio)
    r_h = int(img.shape[0] * ratio)

    i_r = cv2.resize(img, (r_w, r_h), interpolation=cv2.INTER_AREA)

    i_r_p = crop_image_fn.parent.parent / 'crop_resized'
    i_r_p.mkdir(exist_ok=True, parents=False)

    i_r_p = i_r_p / crop_image_fn.name

    if cv2.imwrite(i_r_p, i_r) is False:
      pass
      # need logging