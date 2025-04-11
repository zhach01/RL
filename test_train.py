import os
import numpy as np
import random
import copy

# Allow duplicate OpenMP runtimes (use with caution)
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# Import the controller that integrates environment + agent
from agent import Controller

def main():
    # Print current working directory for debugging.
    cwd = os.getcwd()
    print("Current working directory:", cwd)
    
    # Determine the absolute path to the directory where this script resides.
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Build the absolute path to the trained_models folder.
    trained_models_dir = os.path.join(base_dir, "trained_models")
    
    # Verify that the trained_models directory exists (or create it).
    if not os.path.exists(trained_models_dir):
        print(f"Creating directory for trained models at: {trained_models_dir}")
        os.makedirs(trained_models_dir)

    # Parameters for controller, training, and testing.
    SEED = 4              # Random seed
    NUM_EPISODES = 4000   # Episodes for training
    MAX_TIME_STEPS = 500  # Max timesteps per episode
    MODEL_SAVE_NAME = "reacher"   # Base name used when saving models
    NUM_TESTS = 10
    NUM_EPISODES_CONTINUE = 500  # If continuing training from a prior checkpoint

    # Create the controller with seed, etc.
    controller = Controller(rand_seed=SEED, rew_type=None)

    # Prompt user to choose mode
    mode = input("Enter 'train' to train the model or 'test' to test an existing model: ").strip().lower()

    if mode == 'train':
        # Ask if you want to continue training from an existing model.
        continue_training = input("Continue training from a saved model? (y/n): ").strip().lower()
        continue_model_path = None

        if continue_training == 'y':
            # e.g., "reacher_1000"
            continue_model_name = f"{MODEL_SAVE_NAME}_{NUM_EPISODES_CONTINUE}"
            continue_model_path = os.path.join(trained_models_dir, continue_model_name)
            print("Continuing training from:", continue_model_path)
        
        # Start training
        controller.train(
            num_episodes=NUM_EPISODES,
            max_timesteps=MAX_TIME_STEPS,
            model_name=MODEL_SAVE_NAME,
            continue_from_model=continue_model_path,
            start_episode=NUM_EPISODES_CONTINUE if (continue_training == 'y') else 1
        )

    elif mode == 'test':
        # Reconstruct the model name from training
        model_load_name = f"{MODEL_SAVE_NAME}_{NUM_EPISODES}"
        model_full_path = os.path.join(trained_models_dir, model_load_name)
        print("Loading model from:", model_full_path)

        # Run test
        controller.test(
            num_test=NUM_TESTS,
            max_timesteps=MAX_TIME_STEPS,
            model_name=model_full_path
        )
    else:
        print("Invalid mode selected. Please enter either 'train' or 'test'.")

if __name__ == "__main__":
    main()
