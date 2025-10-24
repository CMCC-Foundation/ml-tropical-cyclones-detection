#!/bin/bash

set -e

# Input parameters
LatRange=${1}
LonRange=${2}
NewGrid=${3}
InFile=${4}
OutFile=${5}

FileName=$InFile
RelWorkDir="`dirname \"$0\"`"
AbsWorkDir="`( cd \"$RelWorkDir\" && pwd )`"

LATS=280
LONS=880

# Bilinear regridding
if [ "$NewGrid" != "" ]; then

XSIZE=${NewGrid%%x*}
XSIZE=${XSIZE##*r}
YSIZE=${NewGrid##*x}
XFIRST=${LonRange%%:*}
YFIRST=${LatRange%%:*}
XLAST=${LonRange##*:}
YLAST=${LatRange##*:}
LATS=`echo "($YLAST)-($YFIRST)" | bc -l`
LONS=`echo "($XLAST)-($XFIRST)" | bc -l`
XINC=`echo "($LONS)/($XSIZE)" | bc -l`
YINC=`echo "($LATS)/($YSIZE)" | bc -l`
let XSIZE+=1
let YSIZE+=1

(
cat <<'EOF'
gridtype = lonlat
xsize = XSIZE
ysize = YSIZE
xfirst = XFIRST
xinc = XINC
yfirst = YFIRST
yinc = YINC
EOF
) > $FileName.grid
sed -i "s/XSIZE/$XSIZE/g" $FileName.grid
sed -i "s/YSIZE/$YSIZE/g" $FileName.grid
sed -i "s/XFIRST/$XFIRST/g" $FileName.grid
sed -i "s/YFIRST/$YFIRST/g" $FileName.grid
sed -i "s/XINC/$XINC/g" $FileName.grid
sed -i "s/YINC/$YINC/g" $FileName.grid

tmp=$FileName.tmp
rm -f $tmp

OUT=0
ncdump -v plev $InFile > /dev/null 2>&1 || OUT=$?
if [ $OUT -eq 0 ]; then
    #ncpdq -a time,plev,lat,lon $InFile $tmp
    ncwa -a plev $InFile $tmp
    rm -f $InFile
    mv $tmp $InFile
    ncks -x -v plev $InFile $tmp
    rm -f $InFile
    mv $tmp $InFile
fi
ncpdq -a time,lat,lon $InFile $tmp
mv $tmp $OutFile

cdo -setctomiss,inf -remapcon,$FileName.grid $OutFile $tmp
mv $tmp $OutFile

rm -f $FileName.grid
rm -f $InFile

ncatted -h -O -a CDO,global,d,, $OutFile
ncatted -h -O -a NCO,global,d,, $OutFile
ncatted -h -O -a history_of_appended_files,global,d,, $OutFile
ncatted -h -O -a history,global,d,, $OutFile

else

if [ "$InFile" != "$OutFile" ]; then
mv $InFile $OutFile
fi

fi

exit 0

