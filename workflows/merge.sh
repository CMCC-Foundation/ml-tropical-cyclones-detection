#!/bin/bash

set -e

Variable=${1}
FirstFile=${2}
SecondFile=${3}
OutputFile=${4}

rm -f $OutputFile
rm -f $OutputFile.tmp

OUT=0
ncdump -v $Variable $FirstFile > /dev/null 2>&1 || OUT=$?
if [ $OUT -eq 0 ]; then
    ncks -A -v $Variable $FirstFile $SecondFile
    mv $SecondFile $OutputFile.tmp
    rm -f $FirstFile
else
    ncks -A -v $Variable $SecondFile $FirstFile
    mv $FirstFile $OutputFile.tmp
    rm -f $SecondFile
fi

# Invert latitude
ncpdq -a -lat $OutputFile.tmp $OutputFile
rm -f $OutputFile.tmp

exit 0
