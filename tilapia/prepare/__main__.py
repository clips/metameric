from argparse import ArgumentParser
from data import process_and_write, FEATURES


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-i",
                        "--input",
                        type=str,
                        required=True)
    parser.add_argument("-o",
                        "--output",
                        type=str,
                        required=True)
    parser.add_argument("-d",
                        "--decomposable",
                        nargs='+')
    parser.add_argument("--decomposable_names",
                        nargs='+')
    parser.add_argument("-f",
                        "--add_features",
                        nargs='+')
    parser.add_argument("--feature_sets",
                        nargs='+')

    args = parser.parse_args()

    feat_names = set(args.feature_names) - set(FEATURES.keys())
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
                      args.output,
                      args.decomposable,
                      args.decomposable_names,
                      args.add_features,
                      args.feature_sets)
