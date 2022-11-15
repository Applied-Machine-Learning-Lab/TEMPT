gpu_id=0

hos_list=(79 141 142 143 144 146 148 154 157 165 167 \
183 184 197 198 199 202 206 208 215 217 224 226 \
227 244 245 248 252 256 259 264 268 269 271 272 275 277 279 280 \
281 282 283 300 301 307 310 312 318 328 331 336 337 338 345 353 365 \
411 413 416 417 419 443 444 449 452 458 459 \
152 92 220 188 181 195 171 110 176 122 420 243 140)
seed_list=(42 43 44 45 46)
for hos_id in ${hos_list[@]}
do
    for seed in ${seed_list[@]}
    do
        python main.py --do_train \
        --num_workers 8 \
        --gpu_id ${gpu_id} \
        --log --use_pretrain \
        --single_train --use_prompt \
        --hos_id ${hos_id} \
        --seed ${seed} \
        --out_exp ./log/results/prompt.json \
        --freeze \
        --pretrain_dir ./saved/eicu/default \
        --check_path normal
    done
done

