rm *.csv
rm *.pdf

cd ../../..

python -m ae.figure5.cf.test_softmax --simgpu --roofline
python -m ae.figure5.cf.test_softmax --simtpu --roofline
python -m ae.figure5.cf.test_softmax --simamd --roofline

python -m ae.figure5.cf.test_softmax --simgpu
python -m ae.figure5.cf.test_softmax --simtpu
python -m ae.figure5.cf.test_softmax --simamd

cd ae/figure5/cf
python plot_softmax.py