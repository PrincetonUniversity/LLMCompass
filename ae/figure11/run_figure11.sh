rm *.csv
rm *.pdf

cd ../..

python -m ae.figure11.test_decoding

cd ae/figure11
python plot_decoding.py