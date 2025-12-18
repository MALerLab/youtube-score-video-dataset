"""
Internal Infos: not relevant for normal users
This file is from lsdp/processors/exclusion_list.py in
https://github.com/MALerLab/lsdp/tree/8899f580b5e556619fdc0edf81c9f1ccc759ef73

ONLY COVERS EXCLUSION LISTS FOR STRING QUARTET VIDEOS
"""

exclude_composers = {
  'string_quartet': {
    'felix_mendelssohn',
  },
}

exclude_pages = {
  'string_quartet': {
    # 240720 last blanks + potraits
    '6pcGvVWWb34:0065', 
    'YlNUmR3RGU8:0175', 
    'alaVnoXoHXY:0167', 
    'gOTqQS5e00k:0141', 
    'BedF8sBSVLI:0164', 
    'ZijJqcfL_Xo:0261', 
    'E1-6_Oor1ak:0147', 
    '1hbqxuoxHJs:0149', 

    # 240828 white blanks
    '5wNGZgvgBW0:0013', 
    '5wNGZgvgBW0:0023', 
    'aCKTEaM9agI:0011', 
    'aCKTEaM9agI:0020', 

    # 240906 black blanks
    'qDfaWaQDCrY:0015', 

    # 240908 transition
    'P0RjfdIBALY:0008', 

    # 240909 flatten pipeline errors
    '0K1x1RK-L14:0067',
    '0K1x1RK-L14:0068',
    '0K1x1RK-L14:0072',
    '0K1x1RK-L14:0073',
    'jJC_qeiOCp4:0061',
    'jJC_qeiOCp4:0063',
    'jJC_qeiOCp4:0069',
    'IVzy7MihxWM:0017',
  },
}