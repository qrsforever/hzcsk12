#!/usr/bin/env bash
# -*- coding:utf-8 -*-


# export PATH=/opt/rh/devtoolset-3/root/usr/bin:$PATH

PYTHON=${PYTHON:-"python"}

echo "Building roi align op..."
cd ops/roi_align
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace

echo "Building roi pool op..."
cd ../roi_pool
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace

echo "Building nms op..."
cd ../nms
make clean
make PYTHON=${PYTHON}

echo "Building dcn..."
cd ../dcn
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace
