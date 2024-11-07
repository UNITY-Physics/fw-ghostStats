#!/bin/bash

# Easiest way is to use the fw api
fw-beta gear config --create
# Add in your API key
# Change the config if you want to use CUDA/CPU

fw-beta gear run -i -e bash

# To execute the run with a specific file, find the hirearchy ID, it gets printed in the beginning of the run on flywheel 
python3 run.py <ID>