Policy Evaluation in Reinforcement Learning
======================
Table of Contents:
- Introduction
- Requirements
- Compiling
- Running
- Code structure
- Contact

--------------------
Introduction
--------------------
This is a framework to examine the efficiency and accuracy of policy evaluation algorithms in reinforcement learning (RL).
In RL, each experiment consists of an agent, an environment, a policy, and a prediction algorithm. In this framework, we have implemented four types of domains: Random MDP, Boyan Chain, Mountain Car, and Energy domain.
Here is a list of implemented algorithms: TD, ATD, EATD, LSTD, T-LSTD, iLSTD, PLSTD.

--------------------
Requirements
--------------------
To use this code you need a compiler for C language. The usual compiler gcc-4 has been tested to work fine. You will need GNU Scientific Library (GSL) for linear algebra operations.

--------------------
Compiling
--------------------
In the root directory, you can modify the "Makefile" before compiling. The first line indicates your compiler command (cc, CC, gcc, ...). On the second line, you can choose your compiler flags (-Wall, ...) depending on what you need and what your compiler provides.
A simple execution of make command can compile the whole code:

$ make

It will take a few seconds to compile the entire project. Depending on your compiler and warning level, it might give some warnings. It should not give any errors though.
After a successful compilation, you can find an executable file called "compare_algorithms" in the root directory.

--------------------
Running
--------------------
Compile first! (see above)

The following is the format of the command with its three mandatory and one optional argument:
compare_algorithms num_runs num_steps mdp_type [mdp_filename]

- num_runs: indicates how many times you want to repeat the experiment so that the results are the average of a few runs. In general, 30 - 100 is a reasonable number.
- num_steps: indicates how many steps you want the agent to perform. The behavior for most of the algorithms doesn't change after a certain step. So a value between 1000 and 5000 should work just fine.
- mdp_type: Put '1' to indicate a random mdp that is generated during the experiment, or '2' for premade mdps. For the latter type, you also need to present mdp_filename. The former does not need it.
- mdp_filename: Only use if mdp_type is '2'. This argument gives the name of the file with the trajectories. The project already provides required files for Boyan, and model mdp.

Example:
$ compare_algorithms 100 2000 1

The above line will run 100 runs of random mdp for 2000 steps.

$ compare_algorithms 30 1000 2 boyan_mdp

This one will run 30 runs of first 1000 steps for boyan chain domain.

--------------------
Code structure
--------------------

The project has three parts: the root directory, algorithms and mdp folders.

``algorithms'':
This folder contains the implementation of the algorithms. Three main categories of algorithms are:

    1. Linear_algorithms:

    2. Matrix_algorithms:

    3. Sketch_algorithms:

On top of these, algorithm_utils contains the functions used in those three files.
The file "algorithm.h" contains a general structure to tune each algorithm and the general functions for initialization, deallocation, reseting and updating. These functions can invoke the correct functions for each file given the structure.

The folder "mdps" contains the codes to work with domains. Each mdp that you are considering has a model, i.e....., like random mdp, you can use the tools from this file.
All the codes for loading the trajectories from a file and working with them are in the trajectory_mdp files.
In the root directory, there are a few files: utils, experiment_utils and compare_algorithm. The file utils contains general computation functions, implemented using gsl.
Experiment_util provides/ensures an interface for the runs and results of each run of the algorithm. It also contains the list of corresponding init and deallocate functions for each algorithm.

Compare_algorithm
is the executable file that runs the whole experiment.
It parses the arguments given in the file, forms a list of algorithms with the desired setting of the parameters and prepares the domain.
Then, it iterates over each algorithm through the list and runs them.
It also measures the running time and calculates the accuracy.
This information is stored in a struct and then in a file. Any other high-level visualization tool can read this file and plot the result(s).


--------------------
Contact
--------------------
For the latest releases, check out the git repository:
https://github.com/marthawhite/MLLab/tree/master/code/prediction

For suggestions, comments and questions, feel free to contact the developers at:
whitem@ualberta.ca
kumarasw@ualberta.ca
