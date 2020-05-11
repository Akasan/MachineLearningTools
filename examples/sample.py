import sys
sys.path.append("../")
from MLTools.DecisionTree import DecisionTree


if __name__ == "__main__":
    dt = DecisionTree()
    dt.set_parameter()
    print(dt.clf)
