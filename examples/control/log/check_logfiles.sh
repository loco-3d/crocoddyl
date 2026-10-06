#!/bin/bash

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

LOGPATH="$( cd "$(dirname "$0")" ; pwd -P )"
cd $LOGPATH

exit_code=0
echo ">>> Checking log files"
for f in *.log ; do
    git diff --quiet "$f"
    if [ $? -eq 0 ]
    then
        echo -e "    $f is ${GREEN}OK${NC}"
    else
        echo -e "${RED}    $f DIFFERS${NC}"
        exit_code=1
    fi
done

if [ $exit_code -eq 0 ] ; then echo ">>> All log files match." ; fi
exit $exit_code
