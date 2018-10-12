"""The page where params are made."""
import os
from argparse import ArgumentParser
from diploria.web.diploria import app


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-p",
                        "--port",
                        default=8080,
                        type=int,
                        help="The port on which to run.")
    parser.add_argument("--host",
                        default="localhost",
                        type=str,
                        help="The host to use.")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    print("Running with {} as local directory.".format(os.getcwd()))
    app.run(host=args.host, port=args.port)
