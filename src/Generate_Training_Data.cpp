/*
 * Blocks.cpp
 * RVO2 Library
 *
 * Copyright 2008 University of North Carolina at Chapel Hill
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please send all bug reports to <geom@cs.unc.edu>.
 *
 * The authors may be contacted via:
 *
 * Jur van den Berg, Stephen J. Guy, Jamie Snape, Ming C. Lin, Dinesh Manocha
 * Dept. of Computer Science
 * 201 S. Columbia St.
 * Frederick P. Brooks, Jr. Computer Science Bldg.
 * Chapel Hill, N.C. 27599-3175
 * United States of America
 *
 * <http://gamma.cs.unc.edu/RVO2/>
 */

/*
 * Based on the Blocks.cpp example file, generate N episodes of RVO2 trajectories
 * with n bots and no static obstacles. Outputs the time-series trajectories to file
 * for use in Supervised Learning of the value function for a learned CA policy
 */

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
#include <ctime>

#if _OPENMP
#include <omp.h>
#endif

#include <RVO.h>

#ifndef M_PI
const float M_PI = 3.14159265358979323846f;
#endif

/* Store the goals of the agents. */
std::vector<RVO::Vector2> goals;

void setupScenario(RVO::RVOSimulator *sim, int n_agents, float dim_world, float dt)
{
	/* Specify the global time step of the simulation. */
	sim->setTimeStep(dt);

	/* Specify the default parameters for agents that are subsequently added.
       { float::neighborDist   , size_t::maxNeighbors, float::timeHorizon, 
         float::timeHorizonObst, float::radius       , float::maxSpeed   , 
	     const::Vector2 &velocity=Vector2() }
	Adjust robot size and maximum speed based on the size of the world */
	sim->setAgentDefaults(15.0f, 10, 5.0f, 5.0f, dim_world/100, dim_world/50);

	/*
	 * Add agents, specifying their start position, and store their goals on the
	 * opposite side of the environment.
	 */

    float range_x =  dim_world;
    float range_y =  dim_world;

    float x, y, xg, yg;

    for (int i = 0; i < n_agents; ++i){
		/* Assign random states and random goals to the agents */
        x = range_x * (((float)rand()/RAND_MAX) - 0.5); xg = range_x * (((float)rand()/RAND_MAX) - 0.5);
        y = range_y * (((float)rand()/RAND_MAX) - 0.5); yg = range_y * (((float)rand()/RAND_MAX) - 0.5);
        sim->addAgent(RVO::Vector2(x, y));
        goals.push_back(RVO::Vector2(xg, yg));
    }
}

void setPreferredVelocities(RVO::RVOSimulator *sim)
{
	/*
	 * Set the preferred velocity to be a vector of unit magnitude (speed) in the
	 * direction of the goal.
	 */
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int i = 0; i < static_cast<int>(sim->getNumAgents()); ++i) {
		RVO::Vector2 goalVector = goals[i] - sim->getAgentPosition(i);

		if (RVO::absSq(goalVector) > 1.0f) {
			goalVector = RVO::normalize(goalVector);
		}

		sim->setAgentPrefVelocity(i, goalVector);

		/*
		 * Perturb a little to avoid deadlocks due to perfect symmetry.
		 */
		float angle = std::rand() * 2.0f * M_PI / RAND_MAX;
		float dist = std::rand() * 0.0001f / RAND_MAX;

		sim->setAgentPrefVelocity(i, sim->getAgentPrefVelocity(i) +
		                          dist * RVO::Vector2(std::cos(angle), std::sin(angle)));
	}
}

bool reachedGoal(RVO::RVOSimulator *sim, float thresh)
{
	/* Check if all agents have reached their goals. */

	for (size_t i = 0; i < sim->getNumAgents(); ++i) {
		if (RVO::absSq(sim->getAgentPosition(i) - goals[i]) > thresh * thresh) {
			return false;
		}
	}

	return true;
}

int main()
{
	/* Set randomizer seed. Fix this number to produce repeatable data */
	std::srand(static_cast<unsigned int>(std::time(NULL)));

	/* Set Simulation Parameters*/
	float dim_world 	= 10; 	// assumes square world with side length dim_world
	float dt 			= 0.25;
	int   episode 		= 0;
	int   n_agents 		= 2;
    int   N_episodes 	= 2;

	/* Data File Row Organization:
	Episode, (px1(t0) py1(t0)) (vx1(t0) vy1(t0)), ..., (pxn(t0) pyn(t0)) (vxn(t0) vyn(t0)),...,
	  		 (px1(tf) py1(tf)) (vx1(tf) vy1(tf)), ..., (pxn(tf) pyn(tf)) (vxn(tf) vyn(tf))
	Each episode maintains it's own row in the file. The number of robots, number of episodes, 
	and simulation dt is written at the top of the file */
	std::ofstream training_data_file("data/training_data.csv");

	/* Write header information */
	training_data_file << n_agents << " Robots" << std::endl;
	training_data_file << N_episodes << " Epsiodes" << std::endl;
	training_data_file << dt << " Delta-t";


    while (episode < N_episodes){

        std::cout << "----- Generating Episode: " << ++episode << " -----" << std::endl;
		training_data_file << std::endl << episode <<", ";

        /* Create a new simulator instance. */
        RVO::RVOSimulator *sim = new RVO::RVOSimulator();

		/* Create vectors to store pos/vel for writing to file*/
		RVO::Vector2 current_pos;
		RVO::Vector2 current_vel;

        /* Set up a randomized scenario. */
        setupScenario(sim, n_agents, dim_world, dt);

        do {

			/* Write the current positions and velocities to file */
			for (int p = 0; p < n_agents; p++){
				current_pos = sim->getAgentPosition(p);
				current_vel = sim->getAgentVelocity(p);
				training_data_file << "("<< current_pos.x() << " " << current_pos.y() <<") (" << current_vel.x() << " " << current_vel.y() <<"), ";
			}

            setPreferredVelocities(sim);
            sim->doStep();
        }
		/* Adjust goal threshold using the size of the world */
        while (!reachedGoal(sim, dim_world/100));

        delete sim;
    }

	/* Write data file */
	training_data_file.close();

	return 0;
}
