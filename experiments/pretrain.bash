gpu_id=0
seed_list=(42 43 44 45 46)

for seed in ${seed_list[@]}
do
    python pretrain.py --num_workers 8 \
    --gpu_id ${gpu_id} \
    --log \
    --seed ${seed} \
    --num_train_epochs 30 \
    --pretrain_dir ./saved/eicu/default \
    --contrast \
    --tau 0.8 \
    --loss_weight 0.01
done
