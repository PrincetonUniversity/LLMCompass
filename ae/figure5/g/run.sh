rm *.csv
rm *.pdf

cd ../../..

python -m ae.figure5.g.test_gelu --simgpu --roofline
python -m ae.figure5.g.test_gelu --simtpu --roofline
python -m ae.figure5.g.test_gelu --simamd --roofline

python -m ae.figure5.g.test_gelu --simgpu
python -m ae.figure5.g.test_gelu --simtpu
python -m ae.figure5.g.test_gelu --simamd

cd ae/figure5/g
python plot_gelu.py