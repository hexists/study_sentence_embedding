#!/usr/bin/env python3.6

import sys
import inspect
from termcolor import colored, cprint

'''
https://stackoverflow.com/a/51343652
https://pypi.org/project/termcolor/
'''

rprint = lambda x: cprint(x, 'red')
gprint = lambda x: cprint(x, 'green')
bprint = lambda x: cprint(x, 'blue')
yprint = lambda x: cprint(x, 'yellow')

rprinte = lambda x: cprint(x, 'red', file=sys.stderr)
gprinte = lambda x: cprint(x, 'green', file=sys.stderr)
bprinte = lambda x: cprint(x, 'blue', file=sys.stderr)
yprinte = lambda x: cprint(x, 'yellow', file=sys.stderr)

def dprint(message='', color=None, file=None):
    callerframerecord = inspect.stack()[1]
    frame = callerframerecord[0]
    info = inspect.getframeinfo(frame)
    # print('[{}:{}:{:04d}]\t{}'.format(info.filename, info.function, info.lineno, message))

    if file == sys.stderr:
        if color == 'red':
            rprinte('[{}:{:04d}]\t{}'.format(info.function, info.lineno, message))
        elif color == 'green':
            gprinte('[{}:{:04d}]\t{}'.format(info.function, info.lineno, message))
        elif color == 'blue':
            bprinte('[{}:{:04d}]\t{}'.format(info.function, info.lineno, message))
        elif color == 'yellow':
            yprinte('[{}:{:04d}]\t{}'.format(info.function, info.lineno, message))
        else:
            print('[{}:{:04d}]\t{}'.format(info.function, info.lineno, message), file=sys.stderr)
    else:
        if color == 'red':
            rprint('[{}:{:04d}]\t{}'.format(info.function, info.lineno, message))
        elif color == 'green':
            gprint('[{}:{:04d}]\t{}'.format(info.function, info.lineno, message))
        elif color == 'blue':
            bprint('[{}:{:04d}]\t{}'.format(info.function, info.lineno, message))
        elif color == 'yellow':
            yprint('[{}:{:04d}]\t{}'.format(info.function, info.lineno, message))
        else:
            print('[{}:{:04d}]\t{}'.format(info.function, info.lineno, message))


def test(msg):
    dprint('== stdout')
    dprint(msg)
    dprint(msg, 'red')
    dprint(msg, 'green')
    dprint(msg, 'blue')
    dprint(msg, 'yellow')

    dprint('== stderr')
    dprint(msg, file=sys.stderr)
    dprint(msg, 'red', file=sys.stderr)
    dprint(msg, 'green', file=sys.stderr)
    dprint(msg, 'blue', file=sys.stderr)
    dprint(msg, 'yellow', file=sys.stderr)


if __name__ == '__main__':
    test('message')
