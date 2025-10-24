#!/bin/bash

set -e

N=${1} # Number of file
Query=${2} # Filter on files (e.g. on time)
pslYear=${3}

base_path=/home/jovyan/data/CMIP6/HighResMIP/CNRM-CERFACS/CNRM-CM6-1-HR/highres-future/r1i1p1f2/6hrPlevPt
target=/home/jovyan/work/vorticity/input
cd $target
rm -f $target/*

for variable in ua va; do
    source=$base_path/$variable/gr/latest
    cd $source
    i=0
    for file in $Query; do
        ln -s $source/$file $target/$file
        echo "Found and linked: $file"
        let "i+=1"
        if [[ "$i" -ge "$N" ]]; then
            break
        fi
    done
done

echo "Current year is ${pslYear}"
source=$base_path/psl/gr/latest
cd $source
for file in ${pslYear}; do
    ln -s $source/$file $target/$file
done

exit 0
