import numpy as np
import pandas as pd

from .component import Component

# Read data using pd.read_csv() - slow, but in system we'll get data in real time

# Read -> Fill empty cells -> save cleaned
#                          -> Apply transforms -> save preprocessed
def fill_empty(data: pd.DataFrame):
    """
    Fill empty cells with values from previous step.
    """
    data = data.ffill()
    return data

def transform_header(data: pd.DataFrame) -> pd.DataFrame:
    """
    data: Датафрейм, считанный из файла конфигуратора
    """

    data = pd.concat([pd.DataFrame([data.columns], columns=data.columns), data],
                     axis=0).reset_index(drop=True)
    # Build header
    date_column = pd.to_datetime(data.iloc[:, 0], errors='coerce', format='%d.%m.%Y %H:%M:%S').dropna().reset_index(drop=True)

    array = data.iloc[0:3, 1::2].to_numpy().astype(str)

    cols = []

    for i in range(array.shape[1]):
        cols.append(array[0, i] + ' ' + array[1, i] + ' ' + array[2, i])

    cols = np.array(cols)

    # Cut bad lines
    delta = data.shape[0] - date_column.shape[0]
    signal_values = data.iloc[delta:, 1::2].reset_index(drop=True)
    signal_values = signal_values.apply(lambda x:
                                        pd.to_numeric(
                                            x.str.replace(',','.'),
                                            errors='coerce')
                                        )

    cols = np.append(['date'], cols)
    signal_values = pd.DataFrame(pd.concat([date_column, signal_values], axis=1).values, columns=cols)

    return signal_values

def split(names: list[str]) -> dict[str, list[str]]:
    name_groups = dict()
    # format of names[i]: name acronym number metric name, join last 2 (or just drop)
    splitted_names = [elem.split() for elem in names]
    for elem in splitted_names:
        acronym, direction, idx = elem[1][:-1], elem[1][-1], elem[2]
        if acronym not in name_groups:
            name_groups[acronym] = [(direction, idx)]
        else:
            name_groups[acronym].append((direction, idx))
    return name_groups

def group(splitted_data: dict[str, list[str]],
          data: pd.DataFrame) -> dict[str, dict[str, list[tuple | np.ndarray]]]:
    last_char = set()
    for key in splitted_data:
        l = len(splitted_data[key])
        last_char.add(key[-1])
        for values in range(l):
            for column in data.columns.to_list():
                if (key + splitted_data[key][values][0] in column) and (splitted_data[key][values][1] in column):
                    splitted_data[key].insert(len(splitted_data[key]), data[column].to_numpy())

    grouped = {k:{} for k in last_char}

    for char in last_char:
        for key in splitted_data:
            if key[-1] == char:
                grouped[char].update({key:splitted_data[key]})

    return grouped
