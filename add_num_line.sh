dir=$PWD/$1
echo "$dir"
parent_dir=$dir/*.txt
for file_dir in $parent_dir
do
    num_of_line=$(wc -l $file_dir | sed 's/^\([0-9]*\).*$/\1/')
    echo $num_of_line > temp.txt
    cat $file_dir >> temp.txt
    rm $file_dir
    mv temp.txt $file_dir
done
echo "Finished!"