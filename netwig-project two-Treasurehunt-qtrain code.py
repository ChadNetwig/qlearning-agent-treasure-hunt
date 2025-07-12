def qtrain(model, maze, **opt):

    # exploration factor
    global epsilon 
    
    # number of epochs
    #n_epoch = opt.get('n_epoch', 15000)
    
    #CLN: correcrted keyname for funtion call arg
    n_epoch = opt.get('epochs', 15000)

    # maximum memory to store episodes
    max_memory = opt.get('max_memory', 1000)

    # maximum data size for training
    data_size = opt.get('data_size', 50)

    # start time
    start_time = datetime.datetime.now()

    # Construct environment/game from numpy array: maze (see above)
    qmaze = TreasureMaze(maze)

    # Initialize experience replay object
    experience = GameExperience(model, max_memory=max_memory)
    
    win_history = []   # history of win/lose game
    hsize = qmaze.maze.size//2   # history window size
    win_rate = 0.0
    
    # pseudocode:
    # For each epoch:
    #    Agent_cell = randomly select a free cell
    #    Reset the maze with agent set to above position
    #    Hint: Review the reset method in the TreasureMaze.py class.
    #    envstate = Environment.current_state
    #    Hint: Review the observe method in the TreasureMaze.py class.
    #    While state is not game over:
    #        previous_envstate = envstate
    #        Action = randomly choose action (left, right, up, down) either by exploration or by exploitation
    #        envstate, reward, game_status = qmaze.act(action)
    #    Hint: Review the act method in the TreasureMaze.py class.
    #        episode = [previous_envstate, action, reward, envstate, game_status]
    #        Store episode in Experience replay object
    #    Hint: Review the remember method in the GameExperience.py class.
    #        Train neural network model and evaluate loss
    #    Hint: Call GameExperience.get_data to retrieve training data (input and target) and pass to model.fit method 
    #          to train the model. You can call model.evaluate to determine loss.
    #    If the win rate is above the threshold and your model passes the completion check, that would be your epoch.
    
    #print('DEBUG: maze =', maze)
    #print('Current epoch is: ', end='')  
    for cur_epoch in range(n_epoch):
        # set vars for current epoch (used in report below)
        loss = 0
        n_episodes = 0
        
        print('DEBUG: current epoch =', cur_epoch)
        # Agent_cell = randomly select a free cell, 2-tuple with range of 0-7 from the constructor's free_cells var
        pirate_cell = random.choice(qmaze.free_cells)
        #print('DEBUG: random pirate_cell =', pirate_cell)
   
        # Reset the maze with agent set to above position (pirate_cell)
        qmaze.reset(pirate_cell)
        
        # envstate = Environment.current_state
        envstate = qmaze.observe()
        #print('DEBUG: envstate', envstate)
        #print('DEBUG: matrix cell value =', maze[pirate_cell[0], pirate_cell[1]])
        
        #valid_actions = qmaze.valid_actions()
        #print('DEBUG: valid_actions', valid_actions)
        
        # While game state is not game over:
        while qmaze.game_status() == 'not_over':
            
            # CLN: 6/14/22 implemented professor Susalla's recommendations (enumerated below):
            # 1. Set valid_actions
            # 2. Break if not a valid action
            # 3. Set prev_envstate
            # 4. Get next action
            # 5. Apply action, get reward and new envstate
            # 6. Store episode (experience)
            # 7. Train neural network model
            # 8. Evaluate win_rate
            
            # 1. set valid actions based on the pirate's current position
            valid_actions = qmaze.valid_actions()
            #print('DEBUG: valid_actions', valid_actions)
            
            # 3. Set prev_envstate
            previous_envstate = envstate
            
            # 4. Get next action
            # action = choose a valid action (left, right, up, down) by random exploration every 1 out of 10 epochs 
            # else, exploit action from experience
            if (cur_epoch % (epsilon * 100)) == 0:
                action = random.choice(valid_actions)
                #print('DEBUG: explore action', action)
            # else exploit
            else:
                action = np.argmax(experience.predict(previous_envstate))
                # ensure predicted action is a valid action
                # 2. Break if predicted action not a valid action
                if action not in valid_actions:
                    break
                else:
                    #print('DEBUG: exploit action', action)
                    pass
           
            # increment n_episodes after taking action
            n_episodes+=1
            
            # 5. Apply action, get reward and new envstate
            # call the act function based on the action to return envstate, reward, and game_status (win, lose, or not_over)
            envstate, reward, game_status = qmaze.act(action)
                      
            # assign the episode data struct
            episode = [previous_envstate, action, reward, envstate, game_status]
            #print('DEBUG: episode =', episode)
            
            # 6. Store episode (experience)
            # Store episode in Experience replay object
            experience.remember(episode)
            
            # get Experience replay data
            inputs, targets = experience.get_data()
            
            # 7. Train neural network model
            # Train neural network with model.fit
            model.fit(inputs, targets, epochs=20, verbose=1)
            
            # evaluate loss
            loss = model.evaluate(inputs, targets)
            #print('loss evaluation = ', loss)
            
            # check for win/lose and update metrics repsectively
            if episode[4] == 'win':
                #print('DEBUG: episode result = ', episode[4])
                win_history.append(1)
                #print('DEBUG: win_history =', win_history)
                win_rate = sum(win_history) / len(win_history)
                print('DEBUG: win_rate=', win_rate)   
                break
            elif episode[4] == 'lose':
                #print('DEBUG: episode result = ', episode[4])
                win_history.append(0)
                win_rate = sum(win_history) / len(win_history)
                break
            else:  
                #print('DEBUG: episode result = ', episode[4])
                pass
         
        # If the win rate is above the threshold and your model passes the completion check, that would be your epoch.
        # 8. Evaluate win_rate
        if win_rate > epsilon and completion_check(model, qmaze):
            print('DEBUG: win rate above epsilon at epoch:', cur_epoch)
    
            #Print the epoch, loss, episodes, win count, and win rate for each epoch
            dt = datetime.datetime.now() - start_time
            t = format_time(dt.total_seconds())
            template = "Epoch: {:03d}/{:d} | Loss: {:.4f} | Episodes: {:d} | Win count: {:d} | Win rate: {:.3f} | time: {}"
            print(template.format(cur_epoch, n_epoch-1, loss, n_episodes, sum(win_history), win_rate, t))
            
            # We simply check if training has exhausted all free cells and if in all
            # cases the agent won.
            if win_rate > 0.9 : epsilon = 0.05

            #TODO: completion_check() is not functioning properly because play_game() called inside of it
            #      is using the trained model but it is not making proper predictions
            #print ('DEBUG: win_history', sum(win_history[-hsize:]), 'hsize', hsize)
            if sum(win_history[-hsize:]) == hsize and completion_check(model, qmaze):
            #if sum(win_history[-hsize:]) == hsize:   
                print("Reached 100%% win rate at epoch: %d" % (cur_epoch))
                break
        
    
    
    # Determine the total time for training
    dt = datetime.datetime.now() - start_time
    seconds = dt.total_seconds()
    t = format_time(seconds)

    #print("n_epoch: %d, max_mem: %d, data: %d, time: %s" % (epoch, max_memory, data_size, t))
    #CLN: Corrected variable name for epoch to n_epoch
    print("\n\nn_epoch: %d, max_mem: %d, data: %d, time: %s" % (n_epoch, max_memory, data_size, t))
    return seconds

# This is a small utility for printing readable time strings:
def format_time(seconds):
    if seconds < 400:
        s = float(seconds)
        return "%.1f seconds" % (s,)
    elif seconds < 4000:
        m = seconds / 60.0
        return "%.2f minutes" % (m,)
    else:
        h = seconds / 3600.0
        return "%.2f hours" % (h,)