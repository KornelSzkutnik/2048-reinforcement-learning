# Main entry point for configuring and starting agent training with chosen hyperparameters.

from td_agent import TDAgent

ALPHA = 0.2
DECAY = 0.8
DECAY_STEP = 1500
LOW_ALPHA_LIMIT = 0.005
NUM_EPISODES = 10000
SAVE_EVERY = 500


if __name__ == "__main__":
    agent = TDAgent(
        name="agent_2048",
        alpha=ALPHA,
        decay=DECAY,
        decay_step=DECAY_STEP,
        low_alpha_limit=LOW_ALPHA_LIMIT,
        with_weights=True,
    )
    agent.train_run(num_eps=NUM_EPISODES, add_weights="already", saving=True)
    print("Training completed!")
