#!/bin/bash
rm -rf build
mkdir build
cd build
cmake ..
make -j
chmod 550 libmindiedump.so
cd ..