import argparse
import logging
import sys

from . import commands
from . import instatrace

parser = argparse.ArgumentParser(description="Cobe control")
parser.add_argument("-b", "--brain", default="cobe.brain")
parser.add_argument("--debug", action="store_true", help=argparse.SUPPRESS)
parser.add_argument("--instatrace", metavar="FILE",
                    help="log performance statistics to FILE")

subparsers = parser.add_subparsers(title="Commands")
commands.ConsoleCommand.add_subparser(subparsers)
commands.InitCommand.add_subparser(subparsers)
commands.LearnCommand.add_subparser(subparsers)
commands.SetStemmerCommand.add_subparser(subparsers)
commands.DelStemmerCommand.add_subparser(subparsers)


def main():
    args = parser.parse_args()

    formatter = logging.Formatter("%(levelname)s: %(message)s")
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    logging.root.addHandler(console)

    if args.debug:
        logging.root.setLevel(logging.DEBUG)
    else:
        logging.root.setLevel(logging.INFO)

    if args.instatrace:
        instatrace.init_trace(args.instatrace)

    args.run(args)

if __name__ == "__main__":
    main()
