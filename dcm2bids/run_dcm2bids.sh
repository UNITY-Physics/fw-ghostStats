#!/bin/bash
#
# Script for converting dicom files to BIDS format and adding the temperature field to the sidecar files
#
# Requirements
# - dcm2bids
# - dcmtk
# - jq
#
# Petter Clemensson
# 
# 2023-03-10

DICOM_DIR="path/to/project/directory/sourcedata/*"
PARTICIPANT_ID="1370004"
CONFIG_FILE="/path/to/project/directory/code/DailyQA_config.json"
OUTPUT_DIR="path/to/project/directory/rawdata/"

# loop over the dicom directories
for DIR in $DICOM_DIR
do 
  # extract the session ID from the dicom directory name
  SESSION_ID=$(echo $DIR | grep -oE "[0-9]{8}")
  # run the function on the directory and session ID
  dcm2bids -d $DIR -p $PARTICIPANT_ID -s $SESSION_ID -c $CONFIG_FILE -o $OUTPUT_DIR
  
  # extract the specific data field using dcmdump
  DICOM_FILE=$(find $DIR -type f -name "*.dcm" | head -n 1)
  TEMP=$(dcmdump +P "0010,4000" $DICOM_FILE | cut -d'[' -f2 | cut -d']' -f1 | grep -oE "[0-9]{2}")

  # loop over the sidecar files in the output directory
  for SIDECAR in $(find $OUTPUT_DIR -type f -name "*.json")
    do
    # update the JSON sidecar file using jq
    jq --arg temp $TEMP '.Temperature = $temp' "${SIDECAR}" > "${SIDECAR}.tmp"
    mv "${SIDECAR}.tmp" "${SIDECAR}"
  done
done