# Cooperative Perception - Mitigating Redundant Messages

## Steps

1- initialize folders:
    a- go to `sumo_map/default` and chose vehicles density file
    b- run prepare_data/initialize_folders.py
2- run all simulations to generate the saved_state folder
3- run calculate_score_mean_std.py to generate the mean and std for each map
4- run message_content_selection.py