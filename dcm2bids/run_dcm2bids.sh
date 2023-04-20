#!/bin/bash
#
# Script for converting dicom files to BIDS format and adding the temperature field to the sidecar files
#
# Requirements
# - dcm2bids
#
# Petter Clemensson
# 
# 2023-03-10

# Notes :)
# - Ensure PROJECT_DIR points to the correct directory
# - PARTICIPANT_ID may NOT include '-' or '_'. Use only numbers and letters to avoid clashes with the BIDS format
# - PARTICIPANT_ID may be site specific, such as phantom number or university initials

PROJECT_DIR="/path/to/project"
PARTICIPANT_ID="LUND"

DICOM_DIR="$PROJECT_DIR/sourcedata/*"
CONFIG_FILE="$PROJECT_DIR/code/dcm2bids/config.json"
OUTPUT_DIR="$PROJECT_DIR/rawdata/"

for DIR in $DICOM_DIR
do
  # extract the session ID from the dicom directory name
  SESSION_ID=$(echo $DIR | grep -oE "[0-9]{8}")
  # run the function on the directory and session ID
  dcm2bids -d $DIR -p $PARTICIPANT_ID -s $SESSION_ID -c $CONFIG_FILE -o $OUTPUT_DIR
  
  # Transfer metadata from DICOM tags in dicom_update_tags.csv to sidecar files
  swoop_update_sidecar $DIR $OUTPUT_DIR/sub-$PARTICIPANT_ID/ses-$SESSION_ID -m dicom_update_tags.csv -v
done