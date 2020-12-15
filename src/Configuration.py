import configparser
import os
from pathlib import Path


class Configuration:
    root_directory = str(Path(__file__).parent.parent)
    configuration_file_name = "settings.ini"
    configuration_file_path = root_directory + "\\" + configuration_file_name

    def __init__(self):
        self.config = configparser.ConfigParser()
        if os.path.isfile(self.configuration_file_path):
            self.config.read(self.configuration_file_path)
        else:
            self.__write_config_file()

    def __write_config_file(self):
        self.config = configparser.ConfigParser()
        self.config['SETTINGS'] = {
            'dataPath': 'D:/weaii/magisterka/semestr2/Analiza i wizualizacja danych/tensorflow_project/grzyby.csv',
            'epochs': '50',
            'testData': 'D:/weaii/magisterka/semestr2/Analiza i wizualizacja danych/tensorflow_project/test_data.csv'
        }
        with open(self.configuration_file_path, 'w') as configFile:
            self.config.write(configFile)

    def get_data_path(self):
        return self.config['SETTINGS']['dataPath']

    def get_epochs(self):
        return int(self.config['SETTINGS']['epochs'])

    def get_test_data_path(self):
        return self.config['SETTINGS']['testData']
