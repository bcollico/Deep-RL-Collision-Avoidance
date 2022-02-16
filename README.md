# AA277 Project
## Deep Reinforcement Learning for Collision Avoidance

## Generating Training Data using RVO2 (ORCA)
ORCA source code was obtained from the [public RVO2 repository](https://github.com/snape/RVO2). The following instructions (and associated Linux commands) can be used to produce the binary executable for generating a set of randomized ORCA trajectories. Ensure that you re-compile after making any changes to the c++ source code.
1. Create and navigate to the build folder
2. Build the RVO2 and data creation script    
3. Compile the code

These steps may be executed in a single line:

     mkdir build && cd build && cmake .. && make
      
The executable will be generated as `build/src/Generate_Training_Data`. Execute the code from the root directory to produce `data/training_data.csv`.
