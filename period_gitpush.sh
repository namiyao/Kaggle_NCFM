#!/bin/bash
#auto git push every 30 minutes

seconds=1800

for i in {1..100}
do
    echo Time $i ---------------------------------------
    date
    aws s3 cp ~/Kaggle_NCFM/model_bbox s3://disneydsy/model_bbox --recursive
    git add .
    git commit -m "auto_period"
    git push
    sleep ${seconds}
done

#nohup  ./period_gitpush.sh &
