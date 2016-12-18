#!/bin/bash


seconds=180

for i in {1..100}
do
    echo Time $i -------------------------------------------------------------------------------------------
    date
    git add .
    git commit -m "auto"
    git push
    sleep ${seconds}
done

#nohup  ./period_gitpush.sh &
