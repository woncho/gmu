in="script1;script2;script3"
list=$(echo $in | tr ";" "\n")
for script in $list
do
	./runscript.sh $script&
done

while true
do
	count = 0
	echo "--- running script list ---"
	for script in $list
	do
		$count=`expr $count + $(ps aux | grep -i $script".sh" | grep -v "grep" | wc -l)`
		ps aux | grep -i $script".sh" | grep -v "grep"
		echo $count
	done
	sleep 1 
done
