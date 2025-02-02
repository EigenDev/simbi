import argparse
import sys

class CustomParser(argparse.ArgumentParser):
    """Class for pretty-printing the help message. """
    command = []
    def error(self, message):
        sys.stderr.write(f'error: {message}\n')
        if "(choose from 'run', 'plot', 'afterglow', 'clone')" in message:
            self.print_help()
        elif 'configurations' not in message:
            if self.command in ['run', 'plot', 'afterglow', 'clone']:
                self.parse_args([self.command, '--help'])
            else:
                self.print_help()
        sys.exit(2)
        
    def parse_args(self, args=None, namespace=None):
        args, argv = super().parse_known_args(args, namespace)
        print(args)
        self.command = args.command
        if argv:
            msg = 'unrecognized arguments: {:s}'
            self.error(msg.format(' '.join(argv)))
        return args