----- synthesizing ESC data

# (1) set number of threads
export JULIA_NUM_THREADS=16

# (2) simulate new ESC-like synthetic data
cd /mbshome/eilbay/deepTX/trainInferSSA
screen -S deepTX_sim
julia --project=. simulation.jl > sim.log 2>&1


------ training

export JULIA_NUM_THREADS=16
screen -S deepTX_train
cd /mbshome/eilbay/deepTX/trainInferSSA
julia --project=. trainSolver.jl > train.log 2>&1

-----

then make changes in 
# nano /mbshome/eilbay/deepTX/trainInferSSA/simulation.jl

sim_size = 2000 #change
using Base.Threads #add
@threads for i = 1:colun_num #add @threads

