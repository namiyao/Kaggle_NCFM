#!/bin/bash
#auto git push every 30 minutes

seconds=1800

for i in {1..100}
do
    echo Time $i ---------------------------------------
    date
    aws s3 cp ~/Kaggle_NCFM/model_bbox/resnet50_FT38_Hybrid_Rep s3://disneydsy/model_bbox/resnet50_FT38_Hybrid_Rep --recursive
    #aws s3 sync ~/Kaggle_NCFM s3://disneydsy/Kaggle_NCFM --exclude "data/*"
    git add .
    git commit -m "auto_period"
    git push
    sleep ${seconds}
done

#nohup  ./period_gitpush.sh &
