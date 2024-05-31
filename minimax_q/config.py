from multiprocessing import cpu_count

lr = 1e-4
eps = 1e-3
grad_norm = 40
batch_size = 64
learning_starts = 50000
save_interval = 500
target_net_update_interval = 2000
gamma = 1.0
prio_exponent = 0.9
importance_sampling_exponent = 0.6

training_steps = 100000
buffer_capacity = 2000000
max_episode_steps = 27000
actor_update_interval = 400
block_length = 100  # cut one episode to numbers of blocks to improve the buffer space utilization

num_actors = 1 # cpu_count()
base_eps = 0.4
alpha = 7
log_interval = 5

# sequence setting
burn_in_steps = 5
learning_steps = 15
forward_steps = 5
seq_len = burn_in_steps + learning_steps + forward_steps

# network setting
hidden_dim = 192

render = False
save_plot = True
test_epsilon = 0.001