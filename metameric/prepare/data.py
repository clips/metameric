"""Prepare word lists for analysis."""
import numpy as np
from copy import deepcopy
from csv import reader, writer
from itertools import chain
from wordkit.features import (fourteen,
                              sixteen,
                              plunkett_phonemes,
                              patpho_bin)


def convert_feature_set(feature_set, negative=True):
    """Converts a feature set to a flat symbolic representation."""
    result = {}
    if isinstance(feature_set, tuple):
        for idx, sub_item in enumerate(feature_set):
            interm = convert_feature_set(sub_item, negative)
            interm = {k: [(x, idx) for x in v]
                      for k, v in interm.items()}
            result.update(interm)
    else:
        for k, v in feature_set.items():
            v = np.array(v)
            pos = np.flatnonzero(v)
            pos = [str(x) for x in pos]
            result[k] = pos
            if negative:
                neg = np.flatnonzero(v - 1)
                neg = ["{}_neg".format(x) for x in neg]
                result[k].extend(neg)

    return result


fourteen[" "] = [0] * 14
sixteen[" "] = [0] * 16
plunkett_phonemes[0][" "] = [0] * 6
plunkett_phonemes[1][" "] = [0] * 6

# TODO: We need some hack to correctly deal with this.
patpho_bin[0]["C"] = [0] * 5
patpho_bin[1]["V"] = [0] * 7

FEATURES = {"fourteen": convert_feature_set(fourteen),
            "sixteen": convert_feature_set(sixteen),
            "plunkett_phonemes": convert_feature_set(plunkett_phonemes),
            "patpho_bin": convert_feature_set(patpho_bin)}

POS_FEATURES = {"fourteen": convert_feature_set(fourteen, False),
                "sixteen": convert_feature_set(sixteen, False),
                "plunkett_phonemes": convert_feature_set(plunkett_phonemes,
                                                         False),
                "patpho_bin": convert_feature_set(patpho_bin, False)}


def read_input_file(f):
    """Read an input file."""
    items = []
    if 'b' in f.mode:
        f = reader((x.decode('utf-8') for x in f))
    else:
        f = reader(f)
    header = next(f)
    for line in f:
        if not line:
            continue
        item = {}
        for k, v in zip(header, line):
            item[k] = v.split()
        items.append(item)

    for k in header:
        if all([len(i[k]) == 1 for i in items]):
            for i in items:
                i[k] = i[k][0]

    return items


def write_file(items, file):
    """Writes an output file."""
    header = list(items[0].keys())
    w = writer(file)
    w.writerow(header)

    for item in items:
        row = []
        for h in header:
            i = item[h]
            if isinstance(i[0], tuple):
                row.append(" ".join(["-".join([str(z) for z in x])
                                     for x in i]))
            elif isinstance(i, tuple):
                row.append(" ".join(i))
            else:
                row.append(i)

        w.writerow(row)


def decompose(items, field, name, length_adaptation=True):
    """Adds letter features to words."""
    items = deepcopy(items)
    max_length = max([max([len(x) for x in item[field]]) for item in items])
    for item in items:
        item[name] = []
        for sub_item in item[field]:
            length = len(sub_item)
            item[name].extend([(l.lower(), idx)
                               for idx, l in enumerate(sub_item)])
            if length_adaptation:
                item[name].extend([(" ", idx)
                                   for idx in range(length, max_length)])

    return items


def add_features(items,
                 feature_set,
                 feature_name='features',
                 field='letters',
                 strict=True):
    """Adds features to words."""
    items = deepcopy(items)
    new_items = []
    for item in items:
        item[feature_name] = []
        for idx, sub_item in enumerate(item[field]):
            try:
                feats = feature_set[sub_item[0]]
            except KeyError as e:
                if not strict:
                    break
                raise e
            item[feature_name].extend([(f, idx) for f in feats])
        else:
            new_items.append(item)

    return new_items


def process_data(items,
                 decomposable=(),
                 decomposable_names=(),
                 feature_layers=(),
                 feature_sets=(),
                 negative_features=True,
                 length_adaptation=True,
                 strict=True):
    """Process data, add fields, and add them to the item."""
    item_keys = set(chain.from_iterable([x.keys() for x in items]))
    if isinstance(decomposable, str):
        decomposable = (decomposable,)
    if isinstance(decomposable_names, str):
        decomposable_names = (decomposable_names,)
    if isinstance(feature_layers, str):
        feature_layers = (feature_layers,)
    if isinstance(feature_sets, str):
        feature_sets = (feature_sets,)
    for x in decomposable:
        if x not in item_keys:
            raise ValueError("Could not decompose '{}', as it was not in the "
                             "set of keys of your items.".format(x))
    if feature_sets and set(feature_sets) - set(FEATURES.keys()):
        raise ValueError("Your feature sets were not one of: {}"
                         "".format(", ".join(FEATURES.keys())))
    for x in feature_layers:
        if x not in item_keys and x not in decomposable_names:
            raise ValueError("Feature layer '{}' was not in your items, but "
                             "also was not a decomposable layer.".format(x))

    if decomposable_names:
        d = zip(decomposable, decomposable_names)
    else:
        d = zip(decomposable, ["{}-decomposed" for x in decomposable])

    for key in decomposable:
        for i in items:
            if isinstance(i[key], str):
                i[key] = (i[key],)

    for field, new_name in d:
        items = decompose(items,
                          field,
                          new_name,
                          length_adaptation)

    for layer_name, name in zip(feature_layers, feature_sets):
        feats = FEATURES[name] if negative_features else POS_FEATURES[name]
        items = add_features(items,
                             feats,
                             '{}-features'.format(layer_name),
                             layer_name,
                             strict=strict)

    return items


def process_and_write(input_file,
                      output_path,
                      decomposable,
                      decomposable_names,
                      feature_layers,
                      feature_sets,
                      strict):
    """Process data and write it to a file."""
    items = read_input_file(input_file)

    items = process_data(items,
                         decomposable,
                         decomposable_names,
                         feature_layers,
                         feature_sets,
                         strict=strict)
    write_file(items, output_path)
