import json


def load_config(filename):
    """ load config file

    Arguments:
    ----------
        filename {str} -- file name of configuration file

    Returns:
        {dict} -- parameter dictionary

    Examples:
    ---------
        >>> data = load_config("../config/DecisionTreeConfig.json")     # Preset parameters for DecisionTree
        >>> data
        {'bootstrap': True, 'class_weight': None, 'criterion': 'gini', 'max_depth': None, 'max_features':
        'auto', 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_sam
        ples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 10, 'n_job
        s': 1, 'oob_score': False, 'random_state': None, 'verbose': 0, 'warm_start': False}
    """
    data = json.load(open(filename, encoding="utf-8"))
    return data


def save_config(param, filename):
    """ save configuration to file

    Arguments:
        param {dict} -- parameter dictionary
        filename {str} -- file name of new configuration file
    """
    json.dump(param, open(filename, "w", encoding="utf-8"), indent=4)