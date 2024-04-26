rm *.csv
rm *.pdf

cd ../..

python -m ae.figure10.test_latency

cd ae/figure10
python plot_latency.py