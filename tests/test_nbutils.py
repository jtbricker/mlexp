"""Tests for `nbutils` package."""
import pytest
from mock import patch

import pandas as pd

from mlexp import nbutils

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