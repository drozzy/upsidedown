Cleaup code
-----------
- ☕️ Get rid of action_fn
- Get rid of initial random rollout
    - 🍃 Make sampling of command work on an empty replay buffer    


Take Additional input as action_prev
-------------------------------------

- 🍃 Create Embedding Layer
- 🍃 Get rid of state' in trajectories
- Rewrite cmd as (dh, dr)
    - Convert do_training to scaling params in NN instead
        - Put x,y on device
            - 🍃 split x into s, dr, dh


Goal item
---------
-- Parent todo item
    -- Child todo item
        🍃 Leaf item
☕️ Completed todo item
    ☕️ Completed sub-item  
🍃 Leaf item