
import pandas as pd

class TextDataset:


    def __init__(self, dataset_path="/Users/marco/Documents/GitHub/EMNLP-MultiPRIDE/script/dataset_input/test_text_class.csv"):

        self.dataset=pd.read_csv(dataset_path, sep=";")


    def print_line(self, index):

        return self.dataset[index]

    #filtra le linee del dataset considerate non classificabili 


    def filter_dataset_notclass(self):

        self.dataset_filter = self.dataset[self.dataset['NOTCLASS']=="NO"].reset_index(drop=True)

