# RRDB
nf = 3
gc = 32

# Super parameters
# eps = 8/255
clamp = 2.0
channels_in = 3
lr = 1e-4
lr2 = 1/255
lr_min = 1e-5
epochs = 5001
weight_decay = 1e-5
init_scale = 0.01


# Super loss

lamda_guide = 1
lamda_low_frequency = 1
lamda_per = 0.001

# Train:

betas = (0.5, 0.999)
weight_step = 200
gamma = 0.9

# Display and logging:
loss_display_cutoff = 2.0  # cut off the loss so the plot isn't ruined
loss_names = ['L', 'lr']
silent = False
live_visualization = False
progress_bar = False
checkpoint_on_error = True
# Load:

pretrain = True

