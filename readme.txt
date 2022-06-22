# Directories：

│   
├── main.py   	# main file
│── model     	# can be used to save the trained models (some pre-trained models' parameters are already placed in for reference)
├── tools	 	# utilities, including the detection algorithms, the training of the networks, and some basic functions for MIMO-OFDM simulations
├── winner_model	# folder for the WINNER-II datasets (should be downloaded from the URL provided at the repository home page)  

The source files for the training of the CG-OAMP-NET are modified from the Onsager deep learning open source code (https://github.com/mborgerding/onsager_deep_learning).



# Python package dependencies
Python>=3.6
Tensorflow>=2.3.0
numpy>=1.18.5
scipy>=1.4.1



# Steps to start
Step 1. Download the source files.

Step 2. Download the WINNER-II model from the provided URL and put the data in the 'winner_model' folder. 

Step 3. Run main.py for the simulation specified by the system configurations in it, including the the MIMO configurations, the OFDM configurations (e.g., whether to use CP), and the used channel model.     



# Other usage descriptions
1. Some pre-trained model parameters are placed at the 'model' folder. To use these parameters, simply copy the file path to the head of the tools/CG_OAMP.py for loading the parameters (otherwise the default parameters of the prototypical algorithms are used).   Naming of the file: algorithm_QAM_Rx_Tx_snr_layers_CP/CP_FREE. 

2. We recommend to compare the running time of the OAMP(-NET) and CG-OAMP(-NET) with the numpy implementations (i.e., tools/OAMP.py and tools/CG_OAMP.py) for convenience.

3. For using the Tensorflow implementations of the detection network, users should train and then test the trained model in one shot of the program run.



# To be done
More detailed comments in the source code and steps to carry out the simulations.