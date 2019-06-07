#!/usr/bin/python

for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
do
	#filename=$(basename "$file" .jpg)
	#echo $filemname
	filename="images/content_images/baseline_vid1_$i.jpg"
	python3 neural_style.py eval --content-image "$filename" --model pytorch_models/epoch_2_Thu_May_16_190329_2019_100000_10000000000.model --output-image output --cuda 1 --style-num 4 --style-id 3 --name "$i"
done
