#!/bin/bash

INSTANCE="$2"
ALGORITHM="$4"
RANDOMSEED="$6"
EPSILON="$8"
HORIZON="${10}"

python3 init.py $INSTANCE $ALGORITHM $RANDOMSEED $EPSILON $HORIZON