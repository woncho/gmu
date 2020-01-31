ps aux|grep "runscript.sh"|grep -v "grep"
pid_list=$(ps aux|grep "runscript.sh"|grep -v "grep"|grep -Eo '^[a-zA-Z0-9]+\s+[0-9]+'|grep -Eo '[0-9]+$')
for pid in $pid_list
do
	kill $pid
	echo "pid($pid) is killed."
done

