#!/bin/bash

# ./bin/run_random_seeds.sh output_prefix training config other_args
#
# e.g.
# ./bin/run_random_seeds.sh dataroot training_config/ner_transformer.json --include-package calypso.token_embedders
#
# will get executed as 
# python -m allennlp.run train training_config/ner_transformer.json --include-package calypso.token_embedders -s dataroot_SEED

# get the command line arguments
output_prefix=$1
shift;

# loop and set the seeds
random_seeds=(1989894904 2294922467 2002866410 1004506748 4076792239)
numpy_seeds=(1053248695 2739105195 1071118652 755056791 3842727116)
pytorch_seeds=(81406405 807621944 3166916287 3467634827 1189731539)

i=0
while [ $i -lt 5 ]; do
    export RANDOM_SEED=${random_seeds[$i]}
    export NUMPY_SEED=${numpy_seeds[$i]}
    export PYTORCH_SEED=${pytorch_seeds[$i]}

    echo "$i"

    outdir=${output_prefix}_SEED_$i
    python -m allennlp.run train $@ -s $outdir
    python -m allennlp.run evaluate $outdir/model.tar.gz /home/swabhas/data/ner_conll2003/eng.testb --cuda-device 0 --output-file $outdir/evaluation_metrics.json

    let i=i+1
done

# compute stats over seeds
python bin/average_random_seeds.py --save_root $output_prefix

