"""Prepare word lists for analysis."""
import numpy as np
from copy import deepcopy
from csv import reader, writer
from wordkit.orthography.features import fourteen, sixteen
from wordkit.phonology.features import plunkett_phonemes, patpho_bin


def convert_feature_set(feature_set, negative=True):
    """Converts a feature set to a flat symbolic representation."""
    result = {}
    if isinstance(feature_set, tuple):
        for idx, sub_item in enumerate(feature_set):
            interm = convert_feature_set(sub_item, negative)
            interm = {k: ["{}-{}".format(x, idx) for x in v]
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
                neg = ["{}-neg".format(x) for x in neg]
                result[k].extend(neg)

    return result


fourteen[" "] = [0] * 14
sixteen[" "] = [0] * 16
plunkett_phonemes[0][" "] = [0] * 6
plunkett_phonemes[1][" "] = [0] * 6

# TODO: We need some hack to correctly deal with this.
patpho_bin[0][" "] = [0] * 5
patpho_bin[1][" "] = [0] * 7

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
    if f.mode.startswith('rb'):
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


def write_file(items, path):
    """Writes an output file."""
    header = list(items[0].keys())
    file = open(path, 'w')
    w = writer(file)
    w.writerow(header)

    for item in items:
        row = [" ".join(item[h]) if isinstance(item[h], (list, tuple, set))
               else item[h] for h in header]
        w.writerow(row)


def decompose(items, field, name, length_adaptation=True):
    """Adds letter features to words."""
    items = deepcopy(items)
    lengths = [len(item[field]) for item in items]
    max_length = max(lengths)
    for item, length in zip(items, lengths):
        for idx, l in enumerate(item[field]):
            item["{}_{}".format(name, idx)] = l.lower()
        if length_adaptation:
            for idx in range(length, max_length):
                item["{}_{}".format(name, idx)] = " "

    return items


def add_features(items,
                 feature_set,
                 feature_name='features',
                 field='orthography',
                 strict=True):
    """Adds features to words."""
    items = deepcopy(items)
    new_items = []
    for item in items:
        for idx, letter in enumerate(item[field]):
            try:
                feats = feature_set[letter.split("-")[0]]
            except KeyError as e:
                if not strict:
                    break
                raise e
            item["{}_{}".format(feature_name, idx)] = feats
        else:
            new_items.append(item)

    return new_items


def process_data(items,
                 decomposable,
                 decomposable_names,
                 feature_layers,
                 feature_sets,
                 negative_features=True,
                 length_adaptation=True,
                 strict=True):
    """Process data, add fields, and add them to the item."""
    if decomposable:
        if decomposable_names:
            d = zip(decomposable, decomposable_names)
        else:
            d = zip(decomposable, ["{}-decomposed" for x in decomposable])

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
