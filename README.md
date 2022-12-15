#Summarization in Small Window Sizes

The following is my implementation for my semester long project.
Each method is implemented in a seperate folder named with the associated method.
BERT = BERT Method
First_n = Article Start
n_then_random = 1-Random Start
Power Law = Power Law
TF-IDF = TF-IDF
All methods are explained in detail in the paper.

All training runs use the trainer class src/training_scripts/trainer.py which provides a wrapper that given a training step function 
will take the gradient and update the parameters for any method.

#Training Baselines, Novel Methods and Experiments
In each folder we have two main scripts, train_[insert_method].py and generate_and_evaluate.py. The main difference between all of the 
different training functions is the data loader were we implement the associated method. Comments have been provided for anything that 
may be unclear. The training .py file implements a training step, validation step, initialization of model, scheduler and optimizer and passes
those to the Trainer class which then runs the training step for the desired amount of steps (4k).

#Evaluating
The generate_and_evaluate.py files are used to write generations to txt files for manual evaluation and record validation statistics and send
them to Weights and Biases which are then reported in the paper. All that is needed is the data, loaded model and validation step which is then passed
to the Evaluation wrapper which runs the validation step for the desired amount of steps (200).

The .sh files are used for running consecutive scripts and contain the default arguments used for each method.

#Data Downloading
Data if not found on your local machine will be automatically downloaded when running the training scripts from the Hugging Face Libary and will
be stored for future use.
