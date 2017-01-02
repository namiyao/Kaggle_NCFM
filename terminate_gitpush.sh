#!/bin/bash
#auto git push before spot instance terminate

seconds=5
status_terminate='"marked-for-termination"'

for i in {1..34560}
do
    status=$(aws ec2 describe-spot-instance-requests --filters "Name=spot-instance-request-id,Values=sfr-c8851d21-89b1-437d-b9ef-8be2c6a37bdf"| jq '.SpotInstanceRequests[0].Status.Code')
    if  [ $status = $status_terminate ]; then
        echo Terminate ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        date
        aws s3 sync ~/Kaggle_NCFM s3://disneydsy/Kaggle_NCFM --recursive --exclude "data/*"
        git add .
        git commit -m "auto_terminate"
        git push
    else
        sleep ${seconds}
    fi
done

#nohup  ./terminate_gitpush.sh &
