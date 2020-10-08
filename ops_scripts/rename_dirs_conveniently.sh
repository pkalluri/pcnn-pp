for file in [0-9]*; 
echo "${file##^[0-9]}";
# do mv "$file" "${file#bla_}";