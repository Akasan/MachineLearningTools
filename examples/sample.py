import sys
sys.path.append("../")
from MLTools.DecisionTree import DecisionTree
from MLTools.DatasetHandler import load_dataset
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    train_X, train_y, test_X, test_y = load_dataset("iris", is_separate=True, is_split_dataset=True)
    dt = DecisionTree()
    dt.set_parameter()
    dt.set_dataset(train_X, train_y, test_X, test_y)
    dt.fit()