#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

tag=ori
n_layer=2
model=sage
wds=anytown
batch=200

lr=1e-2
wd=5e-4

# ours
for idx in {1..5}; do
  for hidden in 16 32 64; do
    for deploy in master dist hydrodist hds hdvar; do
      for obs in 0.05 0.1 0.2; do
        python train_ratio.py --epoch 2000 --adj binary --tag $tag --deploy $deploy --wds $wds --obsrat $obs --batch $batch --run_id $idx --lr $lr --decay $wd --model $model --hidden_dim $hidden --n_layers $n_layer
      done
    done
  done
done

lr=0.2
wd=5e-3

# ours
for idx in {1..5}; do
  for hidden in 16 32 64; do
    for deploy in master dist hydrodist hds hdvar; do
      for obs in 0.05 0.1 0.2; do
        python train_ratio.py --epoch 2000 --adj binary --tag $tag --deploy $deploy --wds $wds --obsrat $obs --batch $batch --run_id $idx --lr $lr --decay $wd --model $model --hidden_dim $hidden --n_layers $n_layer
      done
    done
  done
done

lr=1e-3
wd=1e-5

# ours
for idx in {1..5}; do
  for hidden in 16 32 64; do
    for deploy in master dist hydrodist hds hdvar; do
      for obs in 0.05 0.1 0.2; do
        python train_ratio.py --epoch 2000 --adj binary --tag $tag --deploy $deploy --wds $wds --obsrat $obs --batch $batch --run_id $idx --lr $lr --decay $wd --model $model --hidden_dim $hidden --n_layers $n_layer
      done
    done
  done
done
