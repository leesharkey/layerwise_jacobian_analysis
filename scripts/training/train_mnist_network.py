from lja.managers.training_manager import MnistNetworkTrainingManager

if __name__ == "__main__":
    training_exp = MnistNetworkTrainingManager(enable_training=True)
    training_exp.training_loop()
