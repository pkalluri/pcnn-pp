vertical() {
	if [ $# -eq 1 ]
		then $1 | sed "s/ /\n/g"
	else
		sed "s/ /\n/g"
	fi
}

catch() { 
	out=$($1)
	echo $out
	id=$(echo $out | grep -oP 'Submitted batch job \K\w+')
}

check() {
	if [ $# -eq 1 ]
		then cat save/*$1.out 
	else
        echo $id
		cat save/$id.out
	fi
}

rmv() {
	rm save/$1*
}

### ^^ let's try to make this catch the id automatically
# alias train="sbatch train_all_sbatch.sh" # train
# check() { cat ../save/*$1.out; } # check 202
# id=$(echo $out | tr -dc '0-9')
