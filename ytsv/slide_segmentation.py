"""
Internal Infos: not relevant for normal users
This file is from lsdp/processors/extract_pages.py in
https://github.com/MALerLab/lsdp/tree/8899f580b5e556619fdc0edf81c9f1ccc759ef73
"""

import os
from time import time
from pathlib import Path
import shutil
import re
import logging

import csv
import pandas as pd

import cv2
import numpy as np

from tqdm.auto import tqdm

from moviepy.video.io.VideoFileClip import VideoFileClip
from pydub import AudioSegment, silence

from .utils import format_logger_msg as format_msg



def apply_gaussian_blur(img, amount=5.0):
  return cv2.GaussianBlur(img, (0, 0), amount)


def get_gray_blur(img):
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img = apply_gaussian_blur(img)
  
  return img


def are_diff(img1, img2, threshold=0.0001):
  diff = cv2.absdiff(img1, img2)
  diff = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)[1]
  diff = diff / 255
  
  return np.sum(diff) > threshold


def get_section_list(cap):
  total_frames, fps, width, height = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

  num_pixel = width * height

  cnt = 0

  cur = None
  prev = None

  is_changed = (False, False)

  section_list = []
  cur_section = (0, 0) # (start, type)

  while True:
    ret, frame = cap.read()
    
    # no more frame
    if not ret: 
      cur_section_start, cur_section_type = cur_section
      section_list.append( (cur_section_start, cnt - 1, cur_section_type) )
      break
    
    if cnt == 0:
      prev = frame
      cnt += 1
      continue
    
    if cnt % int(fps // 3) == 0:
      cur = frame
      
      cur_gb = get_gray_blur(cur)
      prev_gb = get_gray_blur(prev)
      
      is_changed = ( is_changed[1], are_diff(cur_gb, prev_gb) )
      
      prev_chng, cur_chng = is_changed
      
      # detect (static end, transition start) or (transition end, static start)
      if (not prev_chng and cur_chng) or (prev_chng and not cur_chng):
        cur_section_start, cur_section_type = cur_section
        section_list.append( (cur_section_start, cnt - 1, cur_section_type) )
        
        cur_section = (cnt, int( not bool(cur_section_type) ))
      
      prev = cur
    
    cnt += 1
  
  return section_list


def get_page_list(sec_ls, total_frames, fps, pad):
  first_section_type = sec_ls[0][2]
  last_section_type = sec_ls[-1][2]

  section_list_sub = sec_ls

  # if first section is transition, drop first
  if first_section_type: 
    section_list_sub = section_list_sub[1:]

  # if last section is transition, drop last
  if last_section_type:
    section_list_sub = section_list_sub[:-1]

  # list of (page_frame, page_start, page_end)
  page_list = []

  for i in range(0, len(section_list_sub), 2):
    section_pair = section_list_sub[i:i+2]
    
    if len(section_pair) < 2:
      break
    
    static, transition = section_pair
    
    page_frame = static[0] + int( (static[1] - static[0]) / 2 )
    page_list.append( (page_frame, max(0, static[0]-pad), min(total_frames, transition[1]+pad) ) )
    
    # overlap = transition[0] + int( (transition[1] - transition[0]) / 2 )
    # page_start = page_list[-1][2] if len(page_list) > 0 else static[0]
    # page_list.append( (page_frame, page_start, overlap) )

  # if len(section_list_sub) is odd num, append last page
  if len(section_list_sub) % 2:
    st, ed, _ = section_list_sub[-1]
    page_frame = st + int( (ed - st) / 2 )
    page_list.append( ( page_frame, max(0, st-pad), min(total_frames, ed+pad) ) )
  
  return page_list


def extract_pages_and_audios(video:VideoFileClip, video_path, out_path, video_id, drop=(True, True)):
  '''
    video_path: Path or str
    out_path: Path or str
    debug: (drop_intro, drop_outro)
  '''
  
  # Paths
  image_out_path = out_path / 'images' / 'original'
  image_out_path.mkdir(parents=True, exist_ok=True)
  
  audio_out_path = out_path / 'audio' / 'original'
  audio_out_path.mkdir(parents=True, exist_ok=True)
  
  # Get video stream 
  cap = cv2.VideoCapture(str(video_path))
  
  total_frames, fps, width, height = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

  # Get sections
  section_list = get_section_list(cap)

  cap.release()
  
  # Get pages
  pad = int(fps / 3) # padding for page duration in frames
  page_list = get_page_list(section_list, total_frames, fps, pad)
  
  # print("page_list : ", page_list)
  
  # Extract audio
  audio = video.audio

  audio_path = audio_out_path / "audio.wav"
  audio.write_audiofile(str(audio_path), logger=None)
  
  audio_segment = AudioSegment.from_wav(audio_path)
  
  
  # Drop silent intros
  num_silence_search_sections = 100
  silence_thresh = -50
  ratio_threshold = 0.6
    
  ratios = []
  
  start_page_idx = 0

  for i, (_, page_start, page_end) in enumerate(page_list):
    # remove pad
    page_start = page_start + pad if i > 0 else page_start
    page_end = page_end - pad if i < len(page_list)-1 else page_end
    
    # convert frame index to seconds
    page_start_time = page_start  / fps
    page_end_time = page_end / fps
    
    # slice audio in milli-second
    page_audio = audio_segment[page_start_time*1000:page_end_time*1000+1]
    
    min_silence_len = len(page_audio) // num_silence_search_sections
    
    segments = silence.detect_nonsilent(page_audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh, seek_step=min_silence_len)
    
    # silent in "%" (how many silent segments / 100)
    seg_dur_sum = sum( [ seg[1] - seg[0] for seg in segments ] )
    ratio = seg_dur_sum / len(page_audio)
    ratios.append((len(page_audio), ratio))
    
    if ratio >= ratio_threshold:
      start_page_idx = i
      break
  
  if drop[0]:
    page_list = page_list[start_page_idx:]
  
  # print("intro non-silent ratios : ", ratios)

  # Drop silent outros
  num_silence_search_sections = 100
  silence_thresh = -50
  ratio_threshold = 0.16
  
  ratios = []
  
  end_page_idx = len(page_list)
  
  for i, (_, page_start, page_end) in reversed(list(enumerate(page_list))):
    # remove pad
    page_start = page_start + pad if i > 0 else page_start
    page_end = page_end - pad if i < len(page_list)-1 else page_end
    
    # convert frame index to seconds
    page_start_time = page_start  / fps
    page_end_time = page_end / fps
    
    # slice audio in milli-second
    page_audio = audio_segment[page_start_time*1000:page_end_time*1000+1]
    
    min_silence_len = len(page_audio) // num_silence_search_sections
    
    if min_silence_len < 1:
      end_page_idx = i
      continue
    
    segments = silence.detect_nonsilent(page_audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh, seek_step=min_silence_len)
    
    # silent in "%" (how many silent segments / 100)
    seg_dur_sum = sum( [ seg[1] - seg[0] for seg in segments ] )
    ratio = seg_dur_sum / len(page_audio)
    ratios.append((len(page_audio), ratio, i))
    
    if ratio >= ratio_threshold:
      break
    
    end_page_idx = i
  
  if drop[1]:
    page_list = page_list[:end_page_idx]
    
  # print("outro non-silent rations : ", ratios)
  # print("num page : ", len(page_list))
  
  # Page extraction and saving image files
  cap = cv2.VideoCapture(str(video_path))

  get_ls_elem = lambda i, ls: ls[i] if i < len(ls) else None

  page_idx = 0
  cur_page = get_ls_elem(page_idx, page_list)

  cnt = 0
  skip_cnt = 0

  change_times = []
  
  while True:
    ret, frame = cap.read()
    
    # no more frame
    if not ret: 
      break

    start_time = cur_page[1] / fps
    end_time = cur_page[2] / fps

    # Write the image of the middle of segment
    if cnt == cur_page[0]:
      cv2.imwrite(str(image_out_path/f'{video_id}:{str(page_idx - skip_cnt).zfill(4)}:{cnt}.png'), frame)
      
      change_times.append((page_idx, skip_cnt, cnt, start_time, end_time))

      page_idx += 1
      cur_page = get_ls_elem(page_idx, page_list)
      
      if not cur_page:
        break
    
    cnt += 1

  cap.release()
  
  # Slice audio and save wav files
  for i in range(len(change_times)):
    page_idx, skip_cnt, cnt, start_time, end_time = change_times[i]

    if i < len(change_times) - 1:
      segment = audio_segment[start_time*1000 : end_time*1000] # time as millisecond
    else:
      # if last segment, add 2 second padding
      segment = audio_segment[start_time*1000 : min(end_time*1000 + (2*1000), len(audio_segment))] # time as millisecond

    segment_name = os.path.join(audio_out_path, f'{video_id}:{str(page_idx - skip_cnt).zfill(4)}:{start_time}:{end_time}.wav')

    segment.export(segment_name, format="wav")

  # Remove the whole audio file
  os.remove(audio_path)


def extract_pages(dataset_dir:Path, yt_metadata:pd.DataFrame, logger:logging.Logger):
  logger_prefix = f'VASeg'
  
  if not isinstance(dataset_dir, Path):
    dataset_dir = Path(dataset_dir)
  
  
  for genre_dir in dataset_dir.iterdir():
    if not genre_dir.is_dir():
      continue
    if len(genre_dir.stem.split('-')) != 2:
      continue
      
    mp4_dir = genre_dir / 'mp4'
    seg_dir = genre_dir / 'segments'

    mp4_paths = sorted(mp4_dir.glob('*.mp4'))
    mp4_paths = list(mp4_paths)
    
    yt_id_pattern = re.compile(r'\[([\w\d_-]+)\]')
    
    # Metadata file setup
    genre_metadata_path = genre_dir / 'metadata-mp4.csv'
    write_mode = 'w'
    
    if genre_metadata_path.exists(): # if previous metadata file exists, append
      write_mode = 'a'
    
    genre_metadata_file = open(genre_metadata_path, write_mode, newline='', encoding='utf-8')
    genre_metadata_writer = csv.writer(genre_metadata_file)
    
    if write_mode == 'w':
      genre_metadata_writer.writerow([
        'genre',
        'yt_id',
        'composer',
        'title',
        'path',
        'video_size',
        'video_frame_rate',
        'video_duration',
        'audio_sample_rate',
        'num_audio_channels',
      ])
    
    for mp4_fn in tqdm(mp4_paths):
      # findall "[SOMETHING]" in title and select last one as yt_id
      yt_id = yt_id_pattern.findall(mp4_fn.stem)[-1]
      yt_seg_dir = seg_dir / yt_id
      
      # if already done, skip extracting
      if yt_seg_dir.exists():
        logger.info(
          format_msg(
            logger_prefix,
            dict(
              status='Pass',
              mp4_name=mp4_fn.stem,
            )
          )
        )
        continue
      
      # extract pages and audios
      start = time()
      try:
        video = VideoFileClip(str(mp4_fn))
        extract_pages_and_audios(video, mp4_fn, yt_seg_dir, yt_id)
        logger.info(
          format_msg(
            logger_prefix,
            dict(
              status='Done in {:.2f} sec'.format(time() - start),
              mp4_name=mp4_fn.stem,
            )
          )
        )
      
      except Exception as e:
        logger.error(
          format_msg(
            logger_prefix,
            dict(
              status=e,
              mp4_name=mp4_fn.stem,
            )
          )
        )
      
      # Extract Metadata for each video
      row = yt_metadata[
        yt_metadata['YT id'] == yt_id
      ]

      composer = row['Composer Full Name'].values[0]
      full_title = row['Title of Video'].values[0]
      
      audio = video.audio
      
      row = (
        genre_dir.stem, 
        yt_id, 
        composer, 
        full_title, 
        '/'.join(mp4_fn.parts[-3:]), # relative path from dataset_dir: <genre>/mp4/<title>.mp4 
        video.size, 
        video.fps, 
        video.duration, 
        audio.fps, 
        audio.nchannels
      )
      
      genre_metadata_writer.writerow(row)
    
    genre_metadata_file.close()