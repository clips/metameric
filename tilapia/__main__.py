import argparse
from tilapia.run import make_run


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Interactive Activation")
    parser.add_argument("-i",
                        "--input",
                        required=True,
                        type=str,
                        help="Path to the input file.")
    parser.add_argument("-t",
                        "--test",
                        help="Path to the test file.")
    parser.add_argument("-o",
                        "--output",
                        required=True,
                        type=str,
                        help="Path to the output file.")
    parser.add_argument("-p",
                        "--parameters",
                        type=str,
                        help="Path to the parameter file.")
    parser.add_argument("-d",
                        "--decay",
                        type=float,
                        default=.07,
                        help="The decay rate.")
    parser.add_argument("-s",
                        "--step",
                        type=float,
                        default=1.,
                        help="The step size.")
    parser.add_argument("--rla",
                        type=float,
                        default=-.05,
                        help="The resting level activation.")
    parser.add_argument("-m",
                        "--min",
                        type=float,
                        default=-.2,
                        help="The minimum activation.")
    parser.add_argument("-S",
                        const=False,
                        default=True,
                        action='store_const',
                        help="Removes space padding.")
    parser.add_argument("-W",
                        const=False,
                        default=True,
                        action='store_const',
                        help="Removes weighting.")
    parser.add_argument("--max_cycles",
                        default=1000,
                        type=int,
                        help="The maximum number of cycles")
    parser.add_argument("-c",
                        "--criterion",
                        default=.7,
                        type=float,
                        help="The threshold for recognition.")
    # look up how this works.
    parser.add_argument("--input_layers",
                        nargs='+',
                        default=('features', 'features_neg'))
    parser.add_argument("--output_layers",
                        nargs='+',
                        default=('orthography',))
    parser.add_argument("--rla_layers",
                        nargs='+',
                        default=('orthography'))
    parser.add_argument("--rla_variable",
                        default="frequency")

    args = parser.parse_args()

    if args.test:
        test = args.test
    else:
        test = args.input

    make_run(args.input,
             test,
             args.output,
             args.parameters,
             args.W,
             args.S,
             args.criterion,
             args.rla,
             args.rla_layers,
             args.input_layers,
             args.output_layers,
             args.rla,
             args.step,
             args.max_cycles,
             args.decay,
             args.min)
