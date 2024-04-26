rm *.csv
rm *.pdf

cd ../../..

python -m ae.figure5.ab.test_matmul --simgpu --roofline
python -m ae.figure5.ab.test_matmul --simtpu --roofline
python -m ae.figure5.ab.test_matmul --simamd --roofline

python -m ae.figure5.ab.test_matmul --simgpu
python -m ae.figure5.ab.test_matmul --simtpu
python -m ae.figure5.ab.test_matmul --simamd

cd ae/figure5/ab
python plot_matmul.py