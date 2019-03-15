"""Tests for `nbutils` package."""
import pytest
from mock import patch

import pandas as pd

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