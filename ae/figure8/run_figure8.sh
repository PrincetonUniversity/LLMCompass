rm *.csv
rm *.pdf

cd ../..

python -m ae.figure8.change_memory_bw

cd ae/figure8
python plot_memory_bw.py