#!/bin/bash
#auto git push before spot instance terminate

seconds=5
status_terminate='"marked-for-termination"'

for i in {1..10000000000000000000000}
do
    status=$(aws ec2 describe-spot-instance-requests --filters "Name=spot-instance-request-id,Values=sir-y7pg4e4k"| jq '.SpotInstanceRequests[0].Status.Code')
    if  [ $status = $status_terminate ]; then
        echo Terminate ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        date
        git add .
        git commit -m "auto_terminate"
        git push
    else
        sleep ${seconds}
    fi
done

#nohup  ./terminate_gitpush.sh &
