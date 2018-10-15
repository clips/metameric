import argparse
from metameric.run import make_run


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Interactive Activation")
    parser.add_argument("-i",
                        "--input",
                        required=True,
                        type=str,
                        help="Path to the training file.")
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
    parser.add_argument("--max_cycles",
                        default=1000,
                        type=int,
                        help="The maximum number of cycles")
    parser.add_argument("--threshold",
                        default=.7,
                        type=float,
                        help="The threshold for recognition.")
    parser.add_argument("--output_layers",
                        nargs='+',
                        default=('orthography',),
                        help="The layers to use as output.")
    parser.add_argument("--monitor_layers",
                        nargs='+',
                        default=('orthography',),
                        help="The layers to monitor for convergence.")
    parser.add_argument("--rla_layers",
                        nargs='+',
                        default=('orthography'),
                        help="The layers on which to perform variable RLA "
                             "weighting.")
    parser.add_argument("--rla_variable",
                        default="frequency",
                        help="The variable on which to base the variable RLA")
    parser.add_argument("-W",
                        const=False,
                        default=True,
                        action="store_const",
                        help="If this switch is passed, weight adaptation"
                             " is not performed")

    args = parser.parse_args()

    if args.test:
        test = args.test
    else:
        test = args.input

    make_run(args.input,
             test,
             args.output,
             args.parameters,
             args.threshold,
             args.rla_variable,
             args.rla_layers,
             args.output_layers,
             args.monitor_layers,
             args.rla,
             args.step,
             args.max_cycles,
             args.decay,
             args.min,
             args.W)
