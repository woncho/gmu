#!/bin/bash
script=$1".sh"
while true
do
rst=`ps aux|grep -i $script|grep -v "grep"|wc -l`
if [ $rst -lt 1 ]
then
	./$script&
	ps aux|grep -i $script|grep -v "grep"
fi
sleep 1
done
