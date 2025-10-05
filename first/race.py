def reward_function(params):
    # example of rewarding the agent to follow center line

    # read input parameters
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']

    # calculate 3 markers that are not varying distances away from the center line
    marker_1 = 0.1 * track_width
    marker_2 = 0.25 * track_width
    marker_3 = 0.5 * track_width

    # give higher reward if the car is closer to the center line and vice versa
    if distance_from_center <= marker_1:
        reward = 1.0
    elif distance_from_center <= marker_2:
        reward = 0.5
    elif distance_from_center <= marker_3:
        reward = 0.1
    else:
        reward = 1e-3  #likely crashed or close to off track

    return float(reward)
# https://student.deepracer.com/createmodel