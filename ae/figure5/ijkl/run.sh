rm *.csv
rm *.pdf

cd ../../..

python -m ae.figure5.ijkl.test_transformer --simgpu --roofline
python -m ae.figure5.ijkl.test_transformer --simtpu --roofline
python -m ae.figure5.ijkl.test_transformer --simgpu --init --roofline
python -m ae.figure5.ijkl.test_transformer --simtpu --init --roofline

python -m ae.figure5.ijkl.test_transformer --simgpu
python -m ae.figure5.ijkl.test_transformer --simtpu
python -m ae.figure5.ijkl.test_transformer --simgpu --init
python -m ae.figure5.ijkl.test_transformer --simtpu --init

cd ae/figure5/ijkl
python plot_transformer.py