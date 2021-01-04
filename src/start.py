import os

from Configuration import Configuration

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import click
import numpy as np
from DataLoader import DataLoader
from Drawing import Drawing
from TensorFlowUtils import TensorFlowUtils

# https://miroslawmamczur.pl/przykladowa-siec-neuronowa-mlp-w-tensorflow/


configuration = Configuration()


@click.group(name="tools")
def cli():
    pass


@cli.command()
def train_model():
    """Rozpoczynanie trenowania modelu"""
    tensor_flow_utils = TensorFlowUtils(DataLoader(configuration.get_data_path()))
    model = tensor_flow_utils.prepare_model(learning_rate=configuration.get_learning_rate(),
                                            algorithm=configuration.get_algorithm())
    tensor_flow_utils.train_model(model, configuration.get_epochs(), configuration.get_batch_size())


@cli.command()
def accuracy_plot():
    """Rysuje wykres pokazujący stopień dokładności przewidywań sieci"""
    drawing = Drawing()
    drawing.draw_accuracy_plot()


@cli.command()
def lose_plot():
    drawing = Drawing()
    drawing.draw_loss_plot()


@cli.command()
def predict():
    """Predykcja klasy na podstawie danych wejściowych w pliku test_data.csv"""
    data_loader = DataLoader(configuration.get_test_data_path(), False)
    tensor_flow_utils = TensorFlowUtils(data_loader)
    model = tensor_flow_utils.prepare_model(learning_rate=configuration.get_learning_rate())
    tensor_flow_utils.load_saved_weights(model)
    predicted_data = tensor_flow_utils.predict(model)
    mapped_predicted_data = np.array(
        list(map(lambda x: [data_loader.class_names[0] + ": " + get_percentage(x[0]) + " " +
                            data_loader.class_names[1] + ": " + get_percentage(x[1])],
                 predicted_data)))
    data_loader.append_predict_column(configuration.get_test_data_path(), mapped_predicted_data)


def get_percentage(number):
    return "{0:.00f}%".format(number * 100)


if __name__ == "__main__":
    cli()
