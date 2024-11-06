#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

model=GAT_RES_Small
wds=anytown
batch=200

lr=5e-4
wd=1e-6

for idx in {1..5}; do
  for deploy in master dist hydrodist hds hdvar; do
    for obs in 0.05 0.1 0.2; do
      for hidden in 32 64; do
        python train_ratio.py --epoch 2000 --adj binary --tag ms --deploy $deploy --wds $wds --obsrat $obs --batch $batch --run_id $idx --lr $lr --decay $wd --model $model --hidden_dim $hidden --n_layers 15
      done
    done
  done
done

model=GAT_RES_Large

for idx in {1..5}; do
  for deploy in master dist hydrodist hds hdvar; do
    for obs in 0.05 0.1 0.2; do
      for hidden in 128 256; do
        python train_ratio.py --epoch 2000 --adj binary --tag ms --deploy $deploy --wds $wds --obsrat $obs --batch $batch --run_id $idx --lr $lr --decay $wd --model $model --hidden_dim $hidden --n_layers 25
      done
    done
  done
done
