from argparse import ArgumentParser
from tilapia.prepare.data import process_and_write, FEATURES


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-i",
                        "--input",
                        type=str,
                        required=True,
                        help="The path to the input data. This should be "
                             "a CSV with a header.")
    parser.add_argument("-o",
                        "--output",
                        type=str,
                        required=True,
                        help="The path to the output data.")
    parser.add_argument("-d",
                        "--decomposable",
                        nargs='+',
                        help="A list of fields in the input csv which can be "
                             "decomposed into smaller units. An example of "
                             "this could be 'orthography', since words can "
                             "decompose into letters.")
    parser.add_argument("--decomposable_names",
                        nargs='+',
                        help="A list of names, the length of -d, which "
                             "specify the names of the decomposed units "
                             "after being decomposed. When the -d flag"
                             "includes 'orthography', this could "
                             "read 'letters', for example.")
    parser.add_argument("-f",
                        "--add_features",
                        nargs='+',
                        help="A list of fields which should be decomposed "
                             "into features. This can apply to fields which "
                             "are not in the original CSV, and which are "
                             "created through decomposition.")
    parser.add_argument("--feature_sets",
                        nargs='+',
                        help="A list of names of feature sets, the length "
                             "of which is equal to the number of fields "
                             "passed to -f. We currently support 2 "
                             "orthographic and 2 phonological feature sets.")
    parser.add_argument("--disable_strict",
                        action='store_const',
                        default=True,
                        const=False,
                        help="If this flag is passed, any words which can "
                             "not be featurized will be deleted. Use with "
                             "caution.")

    args = parser.parse_args()

    feat_names = set(args.feature_sets) - set(FEATURES.keys())
    if feat_names:
        raise ValueError("You passed features that were not in the set of "
                         "offered features: {}".format(feat_names))

    if args.decomposable_names:
        if len(args.decomposable) != len(args.decomposable_names):
            raise ValueError("The number of decomposable items and number"
                             " of names does not match.")
    if args.add_features:
        if len(args.add_features) != len(args.feature_sets):
            raise ValueError("The number of features and number of feature "
                             "names does not match.")

    process_and_write(open(args.input),
                      open(args.output, 'w'),
                      args.decomposable,
                      args.decomposable_names,
                      args.add_features,
                      args.feature_sets,
                      args.disable_strict)
