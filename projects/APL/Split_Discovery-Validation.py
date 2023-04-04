# Copyright 2023 (C) antillia.com. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# 2023/04/04 Copyright (C) antillia.com
# Split_Discovery-Validation.py

import csv
import os
import glob
import shutil
import traceback
from PIL import Image

from distutils.dir_util import copy_tree

def split_discovery_and_validation(master, output_dir, csv_filepath):
  with open(csv_filepath, "r") as f:
    csv_reader = csv.reader(f, delimiter=',')
    line_count = 0
    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)

    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    for row in csv_reader:
        if line_count == 0:
          print(f'Column names are {", ".join(row)}')
          line_count += 1
        else:
          patient_id = row[0]
          diagnosis  = row[1]
          cohort     = row[2]
          line_count += 1
          print(" {} {} {}".format(patient_id, diagnosis, cohort))
          output_subdir = os.path.join(output_dir, cohort + "/" + diagnosis)
          if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)
          src = os.path.join(master, patient_id + "./Signed slides")
          pattern = src + "/*/*.jpg"
          print("---pattern {}".format(pattern))
          image_files = glob.glob(pattern)
          print("---- image_files {}".format(image_files))
          
          for image_file in image_files:
            print("--- image_file {}".format(image_file))
            basename  = os.path.basename(image_file)
            dest_file = os.path.join(output_subdir, basename)
            if not os.path.exists(dest_file):
              print("----copying an image_file {} to {}".format(image_file, output_subdir))
              #input("HHHHT")
              try:
                img = Image.open(image_file)
                cropped_img = img.crop((0, 0, 360, 360))
                cropped_img.save(dest_file, "JPEG", quality = 95)
              except:
                traceback.print_exc() 
            else:
              print("skip copying a file {}".format(image_file))
          #dest = os.path.join(output_subdir, patient_id + "./Signed slides")
       
          #copy_tree(src, dest)

          print(f'Processed {line_count} lines.')  


if __name__ == "__main__":
  master       = "./images_Patient00-105"
  output_dir   = "./AML-API-Images-Patients-Discovery-Validation"
  csv_filepath = "./Images_metadata_table.csv"
  try:
    split_discovery_and_validation(master, output_dir, csv_filepath)
  except:
    traceback.print_exc()
