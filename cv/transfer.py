#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import, division, print_function
import os
import glob
from tqdm import tqdm

orig_modules = os.listdir("cauchy")


def check_lines(each_line):
  line_content = each_line.split(" ")
  # check if is author
  if "author" in each_line.lower():
    return ""
  elif line_content[0].strip() in [
      'from', 'import'
  ] and line_content[1].split(".")[0] in orig_modules:
    line_content[1] = "cauchy." + line_content[1]
    return " ".join(line_content)
  else:
    return each_line


def process_each_file(file_path):
  # read file content
  with open(file_path, "r") as fin:
    lines = fin.readlines()
  if len(lines) > 0:
    lines = [check_lines(_each_line) for _each_line in lines]
    with open(file_path, "w") as fout:
      fout.write("".join(lines))


def add_root():
  orig_files = glob.glob("cauchy/**/*.py", recursive=True)
  for file_path in orig_files:
    process_each_file(file_path)


def main():
  add_root()


if __name__ == "__main__":
  main()