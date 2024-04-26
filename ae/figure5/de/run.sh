rm *.csv
rm *.pdf

cd ../../..

python -m ae.figure5.de.test_layernorm --simgpu --roofline
python -m ae.figure5.de.test_layernorm --simtpu --roofline

python -m ae.figure5.de.test_layernorm --simgpu
python -m ae.figure5.de.test_layernorm --simtpu

cd ae/figure5/de
python plot_layernorm.py