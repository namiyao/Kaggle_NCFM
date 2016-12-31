#!/bin/bash
#auto git push every 30 minutes

seconds=1800

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
