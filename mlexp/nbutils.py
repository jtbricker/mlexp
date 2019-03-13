"""
The :mod:`mlexp.nbutils` module implements severals methods that are used for machine
learning experiments in jupyter notebooks.
"""

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