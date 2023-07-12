### virtual env
## Install package ppo
pip install .

## Train a Model
ppo.train

## Play with a Trained Model
ppo.play


# scripts for test

ppo.server -p_load p_0 -v_load v_0 -p_save fed4_p_1000 -v_save fed4_v_1000 -a 4
ppo.client -n 1000 -p_save 1_p_1000 -v_save 1_v_1000 
ppo.client -n 1000 -p_save 2_p_1000 -v_save 2_v_1000
ppo.client -n 1000 -p_save 3_p_1000 -v_save 3_v_1000
ppo.client -n 1000 -p_save 4_p_1000 -v_save 4_v_1000

ppo.play -p_load fed4_p_1000 -v_load fed4_v_1000 -l fed4-run.log -n 100
ppo.play -p_load 1_p_1000 -v_load 1_v_1000 -l 1-run.log -n 100
ppo.play -p_load 2_p_1000 -v_load 2_v_1000 -l 2-run.log -n 100
ppo.play -p_load 3_p_1000 -v_load 3_v_1000 -l 3-run.log -n 100
ppo.play -p_load 4_p_1000 -v_load 4_v_1000 -l 4-run.log -n 100

ppo.server -p_load p_0 -v_load v_0 -p_save fed4_p_1000 -v_save fed4_v_1000 -a 4 && ppo.play -p_load fed4_p_1000 -v_load fed4_v_1000 -l fed4-run.log -n 100

ppo.client -n 1000 -p_save 1_p_1000 -v_save 1_v_1000 && ppo.play -p_load 1_p_1000 -v_load 1_v_1000 -l 1-run.log -n 100
ppo.client -n 1000 -p_save 2_p_1000 -v_save 2_v_1000 && ppo.play -p_load 2_p_1000 -v_load 2_v_1000 -l 2-run.log -n 100
ppo.client -n 1000 -p_save 3_p_1000 -v_save 3_v_1000 && ppo.play -p_load 3_p_1000 -v_load 3_v_1000 -l 3-run.log -n 100
ppo.client -n 1000 -p_save 4_p_1000 -v_save 4_v_1000 && ppo.play -p_load 4_p_1000 -v_load 4_v_1000 -l 4-run.log -n 100