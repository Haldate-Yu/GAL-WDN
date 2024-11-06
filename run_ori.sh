#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
model=ori
lr=3e-4
wd=1e-6

for idx in {1..10}; do
  for deploy in master dist hydrodist hds hdvar; do
    for obs in 0.05 0.1 0.2; do
      for adj in binary weighted logarithmic; do
        python train_ratio.py --epoch 2000 --adj $adj --tag ms --deploy $deploy --wds anytown --obsrat $obs --batch 200 --run_id $idx --lr $lr --decay $wd --model $model
        python train_ratio.py --epoch 2000 --adj $adj --tag ms --deploy $deploy --wds ctown --obsrat $obs --batch 120 --run_id $idx --lr $lr --decay $wd --model $model
        python train_ratio.py --epoch 2000 --adj $adj --tag ms --deploy $deploy --wds richmond --obsrat $obs --batch 50 --run_id $idx --lr $lr --decay $wd --model $model
      done
    done
  done
done

lr=3e-4
wd=6e-6

for idx in {1..10}; do
  for deploy in master dist hydrodist hds hdvar; do
    for obs in 0.05 0.1 0.2; do
      for adj in binary weighted logarithmic; do
        python train_ratio.py --epoch 2000 --adj $adj --tag ms --deploy $deploy --wds anytown --obsrat $obs --batch 200 --run_id $idx --lr $lr --decay $wd --model $model
        python train_ratio.py --epoch 2000 --adj $adj --tag ms --deploy $deploy --wds ctown --obsrat $obs --batch 120 --run_id $idx --lr $lr --decay $wd --model $model
        python train_ratio.py --epoch 2000 --adj $adj --tag ms --deploy $deploy --wds richmond --obsrat $obs --batch 50 --run_id $idx --lr $lr --decay $wd --model $model
      done
    done
  done
done