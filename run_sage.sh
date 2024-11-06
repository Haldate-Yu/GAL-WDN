#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

tag=ori

lr=1e-2
wd=5e-4

# ours
for idx in {1..5}; do
  for model in sage; do
    for hidden in 32 64 128; do
      for n_layer in 2; do
        for deploy in master dist hydrodist hds hdvar; do
          for obs in 0.05 0.1 0.2; do
            python train_ratio.py --epoch 2000 --adj binary --tag $tag --deploy $deploy --wds anytown --obsrat $obs --batch 200 --run_id $idx --lr $lr --decay $wd --model $model --hidden_dim $hidden --n_layers $n_layer
            python train_ratio.py --epoch 2000 --adj binary --tag $tag --deploy $deploy --wds ctown --obsrat $obs --batch 120 --run_id $idx --lr $lr --decay $wd --model $model --hidden_dim $hidden --n_layers $n_layer
            python train_ratio.py --epoch 2000 --adj binary --tag $tag --deploy $deploy --wds richmond --obsrat $obs --batch 50 --run_id $idx --lr $lr --decay $wd --model $model --hidden_dim $hidden --n_layers $n_layer
          done
        done
      done
    done
  done
done

lr=0.2
wd=5e-3

# ours
for idx in {1..5}; do
  for model in ssgc; do
    for hidden in 32 64 128; do
      for n_layer in 4 8 16; do
        for deploy in master dist hydrodist hds hdvar; do
          for obs in 0.05 0.1 0.2; do
            python train_ratio.py --epoch 2000 --adj binary --tag $tag --deploy $deploy --wds anytown --obsrat $obs --batch 200 --run_id $idx --lr $lr --decay $wd --model $model --hidden_dim $hidden --n_layers $n_layer
            python train_ratio.py --epoch 2000 --adj binary --tag $tag --deploy $deploy --wds ctown --obsrat $obs --batch 120 --run_id $idx --lr $lr --decay $wd --model $model --hidden_dim $hidden --n_layers $n_layer
            python train_ratio.py --epoch 2000 --adj binary --tag $tag --deploy $deploy --wds richmond --obsrat $obs --batch 50 --run_id $idx --lr $lr --decay $wd --model $model --hidden_dim $hidden --n_layers $n_layer
          done
        done
      done
    done
  done
done

lr=1e-3
wd=1e-5

# ours
for idx in {1..5}; do
  for model in ssgc; do
    for hidden in 32 64 128; do
      for n_layer in 4 8 16; do
        for deploy in master dist hydrodist hds hdvar; do
          for obs in 0.05 0.1 0.2; do
            python train_ratio.py --epoch 2000 --adj binary --tag $tag --deploy $deploy --wds anytown --obsrat $obs --batch 200 --run_id $idx --lr $lr --decay $wd --model $model --hidden_dim $hidden --n_layers $n_layer
            python train_ratio.py --epoch 2000 --adj binary --tag $tag --deploy $deploy --wds ctown --obsrat $obs --batch 120 --run_id $idx --lr $lr --decay $wd --model $model --hidden_dim $hidden --n_layers $n_layer
            python train_ratio.py --epoch 2000 --adj binary --tag $tag --deploy $deploy --wds richmond --obsrat $obs --batch 50 --run_id $idx --lr $lr --decay $wd --model $model --hidden_dim $hidden --n_layers $n_layer
          done
        done
      done
    done
  done
done

