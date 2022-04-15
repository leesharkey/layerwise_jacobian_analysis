from lja.managers.training_manager import LogicalNetworkTrainingManager

if __name__ == "__main__":
    training_exp = LogicalNetworkTrainingManager(
        problem="stacked_xor", enable_training=True
    )
    training_exp.training_loop()
