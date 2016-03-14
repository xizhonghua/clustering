#!/bin/bash
echo generating...

mkdir -p build/debug
cd build/debug
cmake -D CMAKE_BUILD_TYPE=Debug ../..
cd -

mkdir -p build/release
cd build/release
cmake -D CMAKE_BUILD_TYPE=Release ../..
cd -

echo "done..."