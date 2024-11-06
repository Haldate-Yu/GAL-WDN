#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

model=m_gcn
lr=1e-5
wd=0

wds=anytown
batch=200
hidden_dim=32
n_layers=5
m_gcn_n_hops=1
m_gcn_n_layers=2

# wds=ctown
# batch=120
# hidden_dim=32
# n_layers=33
# m_gcn_n_hops=2
# m_gcn_n_layers=2

# wds=richmond
# batch=50
# hidden_dim=48
# n_layers=60
# m_gcn_n_hops=3
# m_gcn_n_layers=2


for idx in {1..5}; do
  for deploy in master dist hydrodist hds hdvar; do
    for obs in 0.05 0.1 0.2; do
      python train_ratio.py --epoch 2000 --adj m-GCN --tag ms --deploy $deploy --wds $wds --obsrat $obs --batch $batch --run_id $idx --lr $lr --decay $wd --model $model --hidden_dim $hidden_dim --n_layers $n_layers --m_gcn_n_hops $m_gcn_n_hops  --m_gcn_n_layers $m_gcn_n_layers
    done
  done
done
