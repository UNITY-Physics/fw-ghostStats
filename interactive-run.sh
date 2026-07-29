#!/usr/bin/env bash 

GEAR=fw-ghoststats
IMAGE=flywheel/ghoststats:$1
LOG=ghoststats-$1-$2

echo $IMAGE $LOG

# Command:
docker run -it --rm \
	-v $3/unity/QA/${GEAR}/utils:/flywheel/v0/utils\
	-v $3/unity/QA/${GEAR}/run.py:/flywheel/v0/run.py\
	-v $3/unity/QA/${GEAR}/${LOG}/input:/flywheel/v0/input\
	-v $3/unity/QA/${GEAR}/${LOG}/output:/flywheel/v0/output\
	-v $3/unity/QA/${GEAR}/${LOG}/work:/flywheel/v0/work\
	-v $3/unity/QA/${GEAR}/${LOG}/config.json:/flywheel/v0/config.json\
	$IMAGE
