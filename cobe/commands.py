# Copyright (C) 2014 Peter Teichman

import atexit
import logging
import os
import re
import readline
import Stemmer
import sys
import time

from .brain import Brain

log = logging.getLogger("cobe")


class InitCommand:
    @classmethod
    def add_subparser(cls, parser):
        subparser = parser.add_parser("init", help="Initialize a new brain")

        subparser.add_argument("--force", action="store_true")
        subparser.add_argument("--order", type=int, default=3)
        subparser.add_argument("--megahal", action="store_true",
                               help="Use MegaHAL-compatible tokenizer")
        subparser.set_defaults(run=cls.run)

    @staticmethod
    def run(args):
        filename = args.brain

        if os.path.exists(filename):
            if args.force:
                os.remove(filename)
            else:
                log.error("%s already exists!", filename)
                return

        tokenizer = None
        if args.megahal:
            tokenizer = "MegaHAL"

        Brain.init(filename, order=args.order, tokenizer=tokenizer)


def progress_generator(filename):
    s = os.stat(filename)
    size_left = s.st_size

    fd = open(filename, "rb")
    for line in fd:
        line = line.decode("utf-8", errors="ignore")
        size_left = size_left - len(line)
        progress = 100 * (1. - (float(size_left) / float(s.st_size)))

        yield line, progress

    fd.close()


class LearnCommand:
    @classmethod
    def add_subparser(cls, parser):
        subparser = parser.add_parser("learn", help="Learn a file of text")
        subparser.add_argument("file", nargs="+")
        subparser.set_defaults(run=cls.run)

    @staticmethod
    def run(args):
        b = Brain(args.brain)
        b.start_batch_learning()

        for filename in args.file:
            now = time.time()
            print(filename)

            count = 0
            for line, progress in progress_generator(filename):
                show_progress = ((count % 1000) == 0)

                if show_progress:
                    elapsed = time.time() - now
                    sys.stdout.write("\r%.0f%% (%d/s)" % (progress,
                                                          count / elapsed))
                    sys.stdout.flush()

                b.learn(line.strip())
                count = count + 1

                if (count % 10000) == 0:
                    b.graph.commit()

            elapsed = time.time() - now
            print("\r100%% (%d/s)" % (count / elapsed))

        b.stop_batch_learning()


class ConsoleCommand:
    @classmethod
    def add_subparser(cls, parser):
        subparser = parser.add_parser("console", help="Interactive console")
        subparser.set_defaults(run=cls.run)

    @staticmethod
    def run(args):
        b = Brain(args.brain)

        history = os.path.expanduser("~/.cobe_history")
        try:
            readline.read_history_file(history)
        except IOError:
            pass
        atexit.register(readline.write_history_file, history)

        while True:
            try:
                cmd = input("> ")
            except EOFError:
                print()
                sys.exit(0)

            b.learn(cmd)
            print(b.reply(cmd).encode("utf-8"))


class IrcClientCommand:
    @classmethod
    def add_subparser(cls, parser):
        subparser = parser.add_parser("irc-client",
                                      help="IRC client [requires twisted]")
        subparser.add_argument("-s", "--server", required=True,
                               help="IRC server hostname")
        subparser.add_argument("-p", "--port", type=int, default=6667,
                               help="IRC server port")
        subparser.add_argument("-n", "--nick", default="cobe",
                               help="IRC nick")
        subparser.add_argument("-c", "--channel", required=True,
                               help="IRC channel")
        subparser.add_argument("-l", "--log-channel",
                               help="IRC channel for logging")
        subparser.add_argument("-i", "--ignore-nick", action="append",
                               dest="ignored_nicks",
                               help="Ignore an IRC nick")
        subparser.add_argument("-o", "--only-nick", action="append",
                               dest="only_nicks",
                               help="Only learn from a specific IRC nick")

        subparser.set_defaults(run=cls.run)

    @staticmethod
    def run(args):
        b = Brain(args.brain)

        Runner().run(b, args)


class SetStemmerCommand:
    @classmethod
    def add_subparser(cls, parser):
        subparser = parser.add_parser("set-stemmer",
                                      help="Configure a stemmer")

        subparser.set_defaults(run=cls.run)

        subparser.add_argument("language", choices=Stemmer.algorithms(),
                               help="Stemmer language")

    @staticmethod
    def run(args):
        b = Brain(args.brain)

        b.set_stemmer(args.language)


class DelStemmerCommand:
    @classmethod
    def add_subparser(cls, parser):
        subparser = parser.add_parser("del-stemmer", help="Delete the stemmer")
        subparser.set_defaults(run=cls.run)

    @staticmethod
    def run(args):
        b = Brain(args.brain)

        b.del_stemmer()
