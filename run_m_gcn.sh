#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
model=m_gcn
lr=1e-5
wd=0

for idx in {1..10}; do
  for deploy in master dist hydrodist hds hdvar; do
    for obs in 0.05 0.1 0.2; do
      python train_ratio.py --epoch 2000 --adj m-GCN --tag ms --deploy $deploy --wds anytown --obsrat $obs --batch 200 --run_id $idx --lr $lr --decay $wd --model $model --hidden_dim 32 --n_layers 5 --m_gcn_n_hops 1 --m_gcn_n_layers 2
      python train_ratio.py --epoch 2000 --adj m-GCN --tag ms --deploy $deploy --wds ctown --obsrat $obs --batch 120 --run_id $idx --lr $lr --decay $wd --model $model --hidden_dim 32 --n_layers 33 --m_gcn_n_hops 2 --m_gcn_n_layers 2
      python train_ratio.py --epoch 2000 --adj m-GCN --tag ms --deploy $deploy --wds richmond --obsrat $obs --batch 50 --run_id $idx --lr $lr --decay $wd --model $model --hidden_dim 48 --n_layers 60 --m_gcn_n_hops 3 --m_gcn_n_layers 2
    done
  done
done
