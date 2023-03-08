#!/bin/bash
#
# Script for converting dicom files to BIDS format
#
# Petter Clemensson
# 
# 2020-05-20

DICOM_DIR="/Users/petter/Documents/Work/dcm2bids_second_try/DailyQA/raw_dicom_data/*"
PARTICIPANT_ID="137-0004"
CONFIG_FILE="/Users/petter/Documents/Work/dcm2bids_second_try/DailyQA/code/ghost_config.json"
OUTPUT_DIR="/Users/petter/Documents/Work/dcm2bids_second_try/DailyQA/bids_data/"

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
  for sidecar in $(find $OUTPUT_DIR -type f -name "*.json")
    do
    # update the JSON sidecar file using jq
    jq --arg temp $TEMP '.Temperature = $temp' "${sidecar}" > "${sidecar}.tmp"
    mv "${sidecar}.tmp" "${sidecar}"
  done
done