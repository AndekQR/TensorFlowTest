import configparser
import os
from pathlib import Path

from Algorithm import Algorithm


class Configuration:
    root_directory = str(Path(__file__).parent.parent)
    configuration_file_name = "settings.ini"
    configuration_file_path = root_directory + "\\" + configuration_file_name

    def __init__(self):
        self.config = configparser.ConfigParser(allow_no_value=True)
        if os.path.isfile(self.configuration_file_path):
            self.config.read(self.configuration_file_path)
        else:
            self.__write_config_file()

    def __write_config_file(self):
        self.config.add_section('SETTINGS')
        self.config.set('SETTINGS', 'dataPath',
                        'D:/weaii/magisterka/semestr2/Analiza i wizualizacja danych/tensorflow_project/grzyby.csv')
        self.config.set('SETTINGS', 'testData',
                        'D:/weaii/magisterka/semestr2/Analiza i wizualizacja danych/tensorflow_project/test_data.csv')
        self.config.set('SETTINGS', 'epochs', '50')
        self.config.set('SETTINGS', 'batch_size', '32')
        self.config.set('SETTINGS', 'learning_rate', '0.001')
        self.config.set('SETTINGS', '; levenberg_marquardt or adaptive_moment_estimation')
        self.config.set('SETTINGS', 'algorithm', 'levenberg_marquardt')

        with open(self.configuration_file_path, 'w') as configFile:
            self.config.write(configFile)

    def get_data_path(self):
        return self.config['SETTINGS']['dataPath']

    def get_epochs(self):
        return int(self.config['SETTINGS']['epochs'])

    def get_test_data_path(self):
        return self.config['SETTINGS']['testData']

    def get_batch_size(self):
        return int(self.config['SETTINGS']['batch_size'])

    def get_learning_rate(self):
        return float(self.config['SETTINGS']['learning_rate'])

    def get_algorithm(self) -> Algorithm:
        name = self.config['SETTINGS']['algorithm']
        if name == Algorithm.adaptive_moment_estimation:
            return Algorithm.adaptive_moment_estimation
        elif name == Algorithm.levenberg_marquardt:
            return Algorithm.levenberg_marquardt
