rm *.csv
rm *.pdf

cd ../..

python -m ae.figure7.change_core_size

cd ae/figure7
python plot_core_size.py