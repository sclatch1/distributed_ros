#!/bin/bash

rosnode ping -c1 /rosout > /dev/null 2>&1

if [ $? -eq 0 ]; then
  exit 0
else
  exit 1
fi
