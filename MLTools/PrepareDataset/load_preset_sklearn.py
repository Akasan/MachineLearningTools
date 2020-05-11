from sklearn.datasets import *


__PRESET_LIST = {
    "iris": load_iris,
    "boston": load_boston,
    "breast_cancer": load_breast_cancer,
    "diabetes": load_diabetes,
    "digits": load_digits,
    "files": load_files,
    "linnerud": load_linnerud,
    "wine": load_wine
}


def print_dataset_list():
    """ print dataset list available to load from sklearn.datasets"""
    for k in __PRESET_LIST.keys():
        print(f"- {k}")


def load_preset(name, is_separate_xy=False):
    """ load dataset 

    Arguments:
        name {str} -- dataset name 
                      Please check available name list by calling print_dataset_list

    Keyword Arguments:
        is_separate_xy {bool} -- whether separate data into input values and output values
    """
    if not name in __PRESET_LIST:
        raise Exception(f"{name} dataset is invalid.")

    data = __PRESET_LIST[name]()

    if is_separate_xy:
        return data.data, data.target

    else:
        return data



if __name__ == "__main__":
    data = load_preset(name="iris")
    print(data)
