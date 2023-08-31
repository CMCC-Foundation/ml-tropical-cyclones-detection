#!/bin/bash

# remove python cache directories
for file in `find . -name '__pycache__'`
do
    echo removing file $file
    rm -rf $file
done

# remove DS_Store
for file in `find . -name '.DS_Store'`
do
    echo removing file $file
    rm -rf $file
done
