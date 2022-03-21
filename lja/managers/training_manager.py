import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from lja.managers.manager import Manager
from lja.networks.mlp import NLayerPerceptron
from lja.data_generators.logical_data_generator import LogicalDataGenerator
import os
from datetime import datetime

# import lja.utils.logger # TODO logger


class TrainingManager(Manager):
    def __init__(self):
        super(TrainingManager, self).__init__()
        self.level_of_network = None
        self.type_of_network = None
        self.results_path = None
        self.results_path_session = None
        self.num_epochs = None
        self.batch_size = None
        self.net = None
        self.train_dataset = None
        self.test_dataset = None
        self.loss_func = None
        self.lr = None
        self.momentum = None
        self.dampening = None
        self.weight_decay = None
        self.optimizer = None
        self.train_loader = None
        self.test_loader = None
        self.logger = None
        self.loading_model_name = None

    def preprocess(self, data, labels):
        err_msg = (
            "Instantiate a specific type of TrainingManager " + "to use this method"
        )
        raise NotImplementedError(err_msg)

    def set_up_data_loaders(self):
        train_loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True
        )

        test_loader = torch.utils.data.DataLoader(
            dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False
        )
        return train_loader, test_loader

    def set_up_optimizer(self):
        optimizer = torch.optim.SGD(
            self.net.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            dampening=self.dampening,
            weight_decay=self.weight_decay,
        )
        return optimizer

    def set_up_results_path_session(self):
        # Create session name and directories e.g.
        session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path_session = os.path.join(self.results_path, session_name)
        if not (os.path.exists(results_path_session)):
            os.makedirs(results_path_session)
        return session_name, results_path_session

    def load_model_checkpoint(self, checkpoint_path=None, net=None, optimizer=None):
        if checkpoint_path is not None:
            print("Loading model from %s" % checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            net.load_state_dict(checkpoint["model_state_dict"], self.device)
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("Loaded model from {}.".format(checkpoint_path))
        else:
            print("Using an UNTRAINED model")
        return None

    def save_model_checkpoint(self, checkpoint_path=None):
        torch.save(
            {
                "model_state_dict": self.net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            checkpoint_path,
        )
        print("Model saved to {}".format(checkpoint_path))

    def training_loop(self):
        total_step = len(self.train_loader)
        for epoch in range(self.num_epochs):
            for i, (data, labels) in enumerate(self.train_loader):
                data, labels = self.preprocess(data, labels)

                # Forward pass
                outputs = self.net(data)

                loss = self.loss_func(outputs, labels)

                # Backprop and optimization
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 100.0)

                self.optimizer.step()
                if (i + 1) % self.cfg.print_log_interval == 0:
                    if hasattr(self, "accuracy_metric"):
                        acc = self.accuracy_metric(outputs, labels)
                        print(f"Accuracy: {acc}")
                    print(
                        "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                            epoch + 1, self.num_epochs, i + 1, total_step, loss.item()
                        )
                    )

            # if (epoch + 1) % self.cfg.model_save_interval:
            if (epoch) % self.cfg.model_save_interval == 0:
                model_name = f"model_{epoch:05d}"
                checkpoint_save_path = os.path.join(
                    self.results_path_session, model_name
                )
                self.save_model_checkpoint(checkpoint_save_path)

    def validation_loop(self):
        with torch.no_grad():
            correct = 0
            total = 0
            for data, labels in self.test_loader:
                data, labels = self.preprocess(data, labels)
                outputs = self.net(data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(
                "Accuracy of the network on the 10000 test images: {} %".format(
                    100 * correct / total
                )
            )


class LogicalNetworkTrainingManager(TrainingManager):
    def __init__(self):
        super(LogicalNetworkTrainingManager, self).__init__()
        # FIXME probably turn the below if statements into whole different classes
        self.type_of_network = "logical"
        self.results_path = os.path.join(
            self.cfg.networks.general.training_results_dir, self.type_of_network
        )
        (
            self.session_name,
            self.results_path_session,
        ) = self.set_up_results_path_session()

        self.num_epochs = self.cfg.networks.logical.num_epochs
        self.batch_size = self.cfg.networks.logical.batch_size

        self.net = NLayerPerceptron(
            sizes=self.cfg.networks.logical.sizes,
            last_act=nn.Sigmoid,
        )

        self.train_dataset = LogicalDataGenerator(size=10000)

        self.test_dataset = LogicalDataGenerator(size=256)

        self.loss_func = nn.BCEWithLogitsLoss()
        self.lr = self.cfg.networks.general.lr
        self.momentum = self.cfg.networks.general.momentum
        self.dampening = self.cfg.networks.general.dampening
        self.weight_decay = self.cfg.networks.general.weight_decay
        self.optimizer = self.set_up_optimizer()
        self.train_loader, self.test_loader = self.set_up_data_loaders()

        if self.cfg.networks.logical.load_model_name is not None:
            model_name = self.cfg.networks.logical.load_model_name
            load_checkpoint_path = os.path.join(self.results_path, model_name)
            self.load_model_checkpoint(load_checkpoint_path, self.net, self.optimizer)

    def preprocess(self, input_data, labels):
        input_data = input_data.float().to(self.device)
        labels = labels.float().to(self.device)
        return input_data, labels

    def accuracy_metric(self, outputs, labels):
        acc = (torch.round(outputs) == labels).sum() / torch.ones_like(labels).sum()
        return acc.item()


class MnistNetworkTrainingManager(TrainingManager):
    def __init__(self):
        super(MnistNetworkTrainingManager, self).__init__()
        # FIXME probably turn the below if statements into whole different classes
        self.type_of_network = "mnist"
        self.results_path = os.path.join(
            self.cfg.networks.general.training_results_dir, self.type_of_network
        )
        (
            self.session_name,
            self.results_path_session,
        ) = self.set_up_results_path_session()

        self.num_epochs = self.cfg.networks.mnist.num_epochs
        self.batch_size = self.cfg.networks.mnist.batch_size

        self.net = NLayerPerceptron(
            sizes=self.cfg.networks.mnist.sizes, last_act=nn.Softmax
        )

        self.train_dataset = torchvision.datasets.MNIST(
            root="../data/mnist/",
            train=True,
            transform=transforms.ToTensor(),
            download=True,
        )

        self.test_dataset = torchvision.datasets.MNIST(
            root="../data/mnist/", train=False, transform=transforms.ToTensor()
        )

        self.loss_func = nn.CrossEntropyLoss()
        self.lr = self.cfg.networks.general.lr
        self.momentum = self.cfg.networks.general.momentum
        self.dampening = self.cfg.networks.general.dampening
        self.weight_decay = self.cfg.networks.general.weight_decay
        self.optimizer = self.set_up_optimizer()
        self.train_loader, self.test_loader = self.set_up_data_loaders()

        if self.cfg.networks.mnist.load_model_name is not None:
            model_name = self.cfg.networks.mnist.load_model_name
            load_checkpoint_path = os.path.join(self.results_path, model_name)
            self.load_model_checkpoint(load_checkpoint_path, self.net)

    def preprocess(self, images, labels):
        images = images.reshape(-1, 28 * 28).to(self.device)
        labels = labels.to(self.device)
        return images, labels

    def accuracy_metric(self, outputs, labels):
        acc = (outputs.max(dim=1)[1] == labels).sum() / torch.ones_like(labels).sum()
        return acc.item()
