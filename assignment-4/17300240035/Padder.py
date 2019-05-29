import numpy as np
from fastNLP.core.field import Padder


class AutoPadder_wrapper(Padder):
    """
    根据contents的数据自动判定是否需要做padding。
    (1) 如果元素类型(元素类型是指field中最里层List的元素的数据类型, 可以通过FieldArray.dtype查看，比如['This', 'is', ...]的元素类
        型为np.str, [[1,2], ...]的元素类型为np.int64)的数据不为(np.int64, np.float64)则不会进行padding
    (2) 如果元素类型为(np.int64, np.float64),
        (2.1) 如果该field的内容只有一个，比如为sequence_length, 则不进行padding
        (2.2) 如果该field的内容为List, 那么会将Batch中的List pad为一样长。若该List下还有里层的List需要padding，请使用其它padder。
            如果某个instance中field为[1, 2, 3]，则可以pad； 若为[[1,2], [3,4, ...]]则不能进行pad
    """
    def __init__(self, pad_val=0):
        """
        :param pad_val: int, padding的位置使用该index
        """
        super().__init__(pad_val=pad_val)

    def _is_two_dimension(self, contents):
        """
        判断contents是不是只有两个维度。[[1,2], [3]]是两个维度. [[[1,2], [3, 4, 5]], [[4,5]]]有三个维度
        :param contents:
        :return:
        """
        value = contents[0]
        if isinstance(value , (np.ndarray, list)):
            value = value[0]
            if isinstance(value, (np.ndarray, list)):
                return False
            return True
        return False

    def __call__(self, contents, field_name, field_ele_dtype):
        if not is_iterable(contents[0]):
            array = np.array([content for content in contents], dtype=field_ele_dtype)
        elif field_ele_dtype in (np.int64, np.float64) and self._is_two_dimension(contents):
            max_len = max([len(content) for content in contents])
            array = np.full((len(contents), max_len), self.pad_val, dtype=field_ele_dtype)
            for i, content in enumerate(contents):
                array[i][-len(content):] = content
        else:  # should only be str
            array = np.array([content for content in contents])
        return array

def is_iterable(content):
    try:
        _ = (e for e in content)
    except TypeError:
        return False
    return True