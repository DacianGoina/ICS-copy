'''
This module contains auxiliary algorithms used in different situations.
'''


def check_value_in_list(value, list_v):
    '''
    Check if the given value exists in the given list; the given list is sorted, thus this method use binary search.
    :param value: scalar value
    :param list_v: built-in list
    :return: True, if value exist in list, otherwise False
    '''
    if len(list_v) == 0:
        return False

    left_idx = 0
    right_idx = len(list_v) - 1
    while left_idx <= right_idx:
        mid_idx = (left_idx + right_idx) // 2
        mid_value = list_v[mid_idx]
        if mid_value == value:
            return True

        if value < mid_value:
            right_idx = mid_idx - 1
        else:
            left_idx = mid_idx + 1

    return False