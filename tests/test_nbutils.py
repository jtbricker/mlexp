"""Tests for `nbutils` package."""
import pytest
from mock import patch

import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.datasets import make_classification
from imblearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.linear_model import LogisticRegression

from mlexp import nbutils
from helpers import get_sample_data

def test_reassign_classes_keep_all_data_but_reassign():
    """Return all data but reassign the labels"""
    data = pd.DataFrame([{'class':0,'data':100},{'class':1,'data':200},{'class':0,'data':500}])
    new_data = nbutils.reassign_classes(data, {0:1,1:0}, 'class')

    assert len(data.index) == len(new_data.index)
    assert new_data['class'].value_counts()[0] == data['class'].value_counts()[1]
    assert new_data['class'].value_counts()[1] == data['class'].value_counts()[0]

def test_reassign_classes_keep_only_one_class():
    """Return all data but reassign the labels"""
    data = pd.DataFrame([{'class':0,'data':100},{'class':1,'data':200},{'class':0,'data':500}])
    new_data = nbutils.reassign_classes(data, {0:0}, 'class')

    assert len(new_data.index) == len(data[data['class'] == 0])
    assert new_data['class'].value_counts()[0] == data['class'].value_counts()[0]

def test_reassign_classes_three_groups_into_two():
    """Return all data but reassign the labels"""
    data = pd.DataFrame([{'class':0,'data':100},{'class':1,'data':200},{'class':0,'data':500},{'class':2,'data':400}])
    new_data = nbutils.reassign_classes(data, {0:0, 1:1, 2:1}, 'class')

    assert new_data['class'].value_counts()[0] == data['class'].value_counts()[0]
    assert new_data['class'].value_counts()[1] == data['class'].value_counts()[1] + data['class'].value_counts()[2]

def test_group_classes_calls_reassign_classes():
    with patch('mlexp.nbutils.reassign_classes') as reassign_classes_call:
        data = pd.DataFrame([{'class':0,'data':100},{'class':1,'data':200}])
        groups = {0:0, 1:1}
        nbutils.group_classes(data, groups)
        reassign_classes_call.assert_called_once_with(data, groups, "GroupID")

def test_specificty_all_correct_one():
    """If all predictions are correct, return 1.0"""
    true = [1,0,1,0,1] 
    pred = true

    spec = nbutils.specificity(true, pred)

    assert 1.0 == spec

def test_specificty_all_incorrect_zero():
    """If all predictions are incorrect, return 0.0"""
    true = [1,0,1,0,1] 
    pred = [0,1,0,1,0]

    spec = nbutils.specificity(true, pred)

    assert 0.0 == spec

def test_specificty_correctly_calculates():
    """Calculates specificity correctly"""
    true = [0,0,0,0,0,0,0,0,0,0] 
    pred = [0,0,0,0,1,1,1,1,1,1]

    spec = nbutils.specificity(true, pred)

    assert 0.4 == spec

def test_negative_predictive_value_all_correct_one():
    """If all predictions are correct, return 1.0"""
    true = [1,0,1,0,1] 
    pred = true

    npv = nbutils.negative_predictive_value(true, pred)

    assert 1.0 == npv

def test_negative_predictive_value_all_incorrect_zero():
    """If all predictions are incorrect, return 0.0"""
    true = [1,0,1,0,1] 
    pred = [0,1,0,1,0]

    npv = nbutils.negative_predictive_value(true, pred)

    assert 0.0 == npv

def test_negative_predictive_value_correctly_calculates():
    """Calculates npv correctly"""
    true = [0,0,0,0,1,1,1,1,1,1]
    pred = [0,0,0,0,0,0,0,0,0,0] 

    npv = nbutils.negative_predictive_value(true, pred)

    assert 0.4 == npv

def test_get_weighted_confusion_matrix_correct():
    """Calculates weighted confusion matrix correctly"""
    true, pred = get_sample_data(135,53,2,11)

    tp, fp, fn, tn = nbutils.get_weighted_confusion_matrix(true, pred)

    assert tp == pytest.approx(0.9247, 0.1)
    assert tn == pytest.approx(0.9636, 0.1)
    assert fp == pytest.approx(0.0364, 0.1)
    assert fn == pytest.approx(0.07534, 0.1)

def test_weighted_accuracy_correct():
    """Calculates weighted accuracy correctly"""
    true, pred = get_sample_data(135,53,2,11)

    w_acc = nbutils.weighted_accuracy(true, pred)

    assert w_acc == pytest.approx(0.94414, 0.1)

def test_weighted_sensitivity_correct():
    """Calculates weighted sensitivity correctly"""
    true, pred = get_sample_data(135,53,2,11)

    w_acc = nbutils.weighted_sensitivity(true, pred)

    assert w_acc == pytest.approx(0.924657, 0.1)

def test_weighted_specificity_correct():
    """Calculates weighted specificity correctly"""
    true, pred = get_sample_data(135,53,2,11)

    w_acc = nbutils.weighted_specificity(true, pred)

    assert w_acc == pytest.approx(0.924657, 0.1)

def test_weighted_ppv_correct():
    """Calculates weighted postiive predictive value correctly"""
    true, pred = get_sample_data(135,53,2,11)

    w_acc = nbutils.weighted_ppv(true, pred)

    assert w_acc == pytest.approx(0.924657, 0.1)

def test_weighted_npv_correct():
    """Calculates weighted negative predictive value correctly"""
    true, pred = get_sample_data(135,53,2,11)

    w_acc = nbutils.weighted_npv(true, pred)

    assert w_acc == pytest.approx(0.924657, 0.1)

def test_print_score_summaries_three_scores_three_summaries(capsys):
    """If there are three score sets in input, print three summaries"""
    scores_dict = {'average': np.array([1,0,1,0]), 'ppv': np.array([1,1,1,1]), 'npv':np.array([0,0,0,0])}
    
    nbutils.print_score_summaries(scores_dict)

    captured = capsys.readouterr()
    assert captured.out.count("\n") == 3

def test_print_score_summaries_ignores_nan(capsys):
    """If there are nan values in score array, ignore in summary calculation"""
    scores_dict = {'average': np.array([1,1,1,1, np.nan, np.nan])}
    
    nbutils.print_score_summaries(scores_dict)

    captured = capsys.readouterr()
    assert "1.0\t0.0" in captured.out 

def test_get_metrics_correctly_calculates_scores():
    """ If a scoring list is not provided use the default value """
    X = [[1,2], [2,2], [3,2]]
    y = [0,1,1]
    model = DummyClassifier(strategy="constant", constant=1).fit(X,y)
    scoring_list = {'average':make_scorer(accuracy_score), 'average2':make_scorer(accuracy_score)}

    metrics = nbutils.get_metrics(model, X, y, scoring_list)

    assert 'average' in metrics
    assert 'average2' in metrics

def test_plot_roc_verbose_true_prints_extra_data(capsys):
    """ If verbose flag is true, print extra optional information """
    X = [[1,2], [2,2], [3,2]]
    y = [1,1,0]
    model = DummyClassifier(strategy="constant", constant=1).fit(X,y)

    nbutils.plot_roc(model, X, y, verbose=True, show_plot=True)

    captured = capsys.readouterr()
    assert "CLASSIFICATION" in captured.out 
    assert "PROBABILITIES" in captured.out 
    assert "RAW DATA" in captured.out

def test_plot_coefficients_no_errors():
    """ Shows plot without error """
    X = [[1,2], [2,2], [3,2]]
    y = [1,1,0]
    model = LogisticRegression().fit(X,y)

    nbutils.plot_coefficients(model, ['first','second'], 1, show_plot=True)

def test_print_feature_importance_different_length_inputs_assertion_error():
    names = ["one", "two", "three"]
    coefs = [1.0, 2.0]

    with pytest.raises(AssertionError):
        nbutils.print_feature_importance(names, coefs)

def test_print_feature_importance_same_length_inputs_no_error():
    names = ["one", "two", "three"]
    coefs = [1.0, 2.0, 3.0]

    nbutils.print_feature_importance(names, coefs)

def test_plot_confusion_matrix_no_errors():
    """ Shows plot without error """
    confusion_matrix = np.array([[10,1],[3,17]])

    nbutils.plot_confusion_matrix(confusion_matrix, show_plot=True, print_matrix=True)

def test_grid_search_optimization_no_errors():
    """ Runs optimization without error """
    param_grid = {}
    clf = Pipeline([('classifier', LogisticRegression())])
    X, y  = make_classification()

    nbutils.grid_search_optimization(clf, param_grid, X, y, X, y, cv=2, n_jobs=1, verbose=True)