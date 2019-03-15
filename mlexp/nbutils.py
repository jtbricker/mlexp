"""
The :mod:`mlexp.nbutils` module implements severals methods that are used for machine
learning experiments in jupyter notebooks.
"""
from sklearn.metrics import confusion_matrix

def group_classes(data, grouping):
    """ Wrapper for backwards compatibility. See :func:`<nbutils.reassign_classes>`"""
    return reassign_classes(data, grouping, 'GroupID')

def reassign_classes(data, grouping, group_col):
        """
        Returns a subset of the data with new class labels
        ----------
        data : DataFrame
        grouping : dict, keys = classes to keep, values = new labels of classes
        Returns
        -------
        data_subset : DataFrame
            subset of data, where only rows with class labels that are keys in grouping 
            are kept and whose new class labels are the corresponding values in grouping
        """
        classes_to_keep = grouping.keys()
        data_to_keep = data.loc[data[group_col].isin(classes_to_keep)]
        classes_to_change = {k:grouping[k] for k in classes_to_keep if k!= grouping[k]}
        return data_to_keep.replace(classes_to_change)

def specificity(y_true, y_pred):
    """ Calculates the specificity (Selectivity, True Negative Rate)
    
    Arguments:
        y_true {array-like} -- true classes
        y_pred {array-like} -- predicted classes
    
    Returns:
        float -- specificity
    """

    cm = confusion_matrix(y_true, y_pred)
    return cm[0,0] / cm[0,:].sum()

def negative_predictive_value(y_true, y_pred):
    """ Calculates the negative predictive value
    
    Arguments:
        y_true {array-like} -- true classes
        y_pred {array-like} -- predicted classes
    
    Returns:
        float -- negative_predictive_value
    """

    cm = confusion_matrix(y_true, y_pred)
    return cm[0,0] / cm[:,0].sum()

def get_weighted_confusion_matrix(y_true, y_pred):
    """ Calculates the confusion matrix weighted by
    class size
    
    Arguments:
        y_true {array-like} -- true classes
        y_pred {array-like} -- predicted classes
    
    Returns:
        tp_weighted -- weighted true positives
        fp_weighted -- weighted false positives
        fn_weighted -- weighted false negatives
        tn_weighted -- weighted true negatives
    """

    cm = confusion_matrix(y_true, y_pred)
    tn = cm[0,0]
    tp = cm[1,1]
    fp = cm[0,1]
    fn = cm[1,0]
    
    tp_weighted = tp / (tp + fn)
    fp_weighted = fp / (tn + fp)
    fn_weighted = fn / (tp + fn)
    tn_weighted = tn / (tn + fp)
    
    return tp_weighted, fp_weighted, fn_weighted, tn_weighted