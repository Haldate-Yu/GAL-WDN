#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
model=GAT_RES_Small
lr=5e-4
wd=1e-6

for idx in {1..10}; do
  for deploy in master dist hydrodist hds hdvar; do
    for obs in 0.05 0.1 0.2; do
      for hidden in 32 64; do
        python train_ratio.py --epoch 2000 --adj binary --tag ms --deploy $deploy --wds anytown --obsrat $obs --batch 200 --run_id $idx --lr $lr --decay $wd --model $model --hidden_dim $hidden --n_layers 15
        python train_ratio.py --epoch 2000 --adj binary --tag ms --deploy $deploy --wds ctown --obsrat $obs --batch 120 --run_id $idx --lr $lr --decay $wd --model $model --hidden_dim $hidden --n_layers 15
        python train_ratio.py --epoch 2000 --adj binary --tag ms --deploy $deploy --wds richmond --obsrat $obs --batch 50 --run_id $idx --lr $lr --decay $wd --model $model --hidden_dim $hidden --n_layers 15
      done
    done
  done
done

model=GAT_RES_Large

for idx in {1..10}; do
  for deploy in master dist hydrodist hds hdvar; do
    for obs in 0.05 0.1 0.2; do
      for hidden in 128 256; do
        python train_ratio.py --epoch 2000 --adj binary --tag ms --deploy $deploy --wds anytown --obsrat $obs --batch 200 --run_id $idx --lr $lr --decay $wd --model $model --hidden_dim $hidden --n_layers 25
        python train_ratio.py --epoch 2000 --adj binary --tag ms --deploy $deploy --wds ctown --obsrat $obs --batch 120 --run_id $idx --lr $lr --decay $wd --model $model --hidden_dim $hidden --n_layers 25
        python train_ratio.py --epoch 2000 --adj binary --tag ms --deploy $deploy --wds richmond --obsrat $obs --batch 50 --run_id $idx --lr $lr --decay $wd --model $model --hidden_dim $hidden --n_layers 25
      done
    done
  done
done
