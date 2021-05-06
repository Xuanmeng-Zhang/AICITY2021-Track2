cd scene1
python split_scene1_res.py
cd ..

cd scene2
python split_scene2_res.py
python get_final_res.py --txt_name track2.txt
