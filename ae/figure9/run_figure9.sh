rm *.csv
rm *.pdf

cd ../..

python -m ae.figure9.change_l1_cache

cd ae/figure9
python plot_l1_cache.py