rm -rf anomalytransfer/out/*
cd sample/scripts/clustering/
python step1_preprocessing.py Yahoo
python step2_baseline_extraction.py Yahoo
python step3_average.py Yahoo
python step4_clustering.py Yahoo
cd ../transfer_entirely/
python finetune3.py Yahoo
cd ../../../


rm -rf anomalytransfer/out/*
cd sample/scripts/clustering/
python step1_preprocessing.py NAB
python step2_baseline_extraction.py NAB
python step3_average.py NAB
python step4_clustering.py NAB
cd ../transfer_entirely/
python finetune3.py NAB
cd ../../../


rm -rf anomalytransfer/out/*
cd sample/scripts/clustering/
python step1_preprocessing.py WSD
python step2_baseline_extraction.py WSD
python step3_average.py WSD
python step4_clustering.py WSD
cd ../transfer_entirely/
python finetune3.py WSD
cd ../../../

# rm -rf anomalytransfer/out/*
# python /sample/scripts/clustering/step1_preprocessing.py
# python /sample/scripts/clustering/step2_baseline_extraction.py
# python /sample/scripts/clustering/step3_average.py
# python /sample/scripts/clustering/step4_clustering.py
# python /sample/scripts/transfer_entirely/finetune3.py WSD

# rm -rf anomalytransfer/out/*
# python /sample/scripts/clustering/step1_preprocessing.py
# python /sample/scripts/clustering/step2_baseline_extraction.py
# python /sample/scripts/clustering/step3_average.py
# python /sample/scripts/clustering/step4_clustering.py
# python /sample/scripts/transfer_entirely/finetune3.py AIOPS


