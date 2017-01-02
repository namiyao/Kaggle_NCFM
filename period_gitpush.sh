#!/bin/bash
#auto git push every 30 minutes

seconds=1800

for i in {1..100}
do
    echo Time $i ---------------------------------------
    date
    aws s3 sync ~/Kaggle_NCFM s3://disneydsy/Kaggle_NCFM --recursive --exclude "data/*"
    git add .
    git commit -m "auto_period"
    git push
    sleep ${seconds}
done

#nohup  ./period_gitpush.sh &
