#!/bin/bash

# remove unnecessary files
rm -rf docs demos imagesite

# create new package
mkdir cauchy
mv datasets extensions methods metrics models utils cauchy
cp setup.py cauchy

# reformat with yapf
# yapf --style='{based_on_style: chromium, indent_width: 2}' -ir **/*.py

# add cauchy head
# python add head 

# change nms make file