"""
Source-preserving reconstruction
=================================

Demonstrates PreservingReconstructor: edit a parse tree and get back text
where only the edited parts are regenerated. Whitespace, blank lines and
%ignored comments of the untouched parts are preserved byte-for-byte,
unlike lark.reconstruct.Reconstructor, which re-renders the whole tree.

Requires an unambiguous LALR parser with propagate_positions=True.
"""

from lark import Lark, Token
from lark.preserve import PreservingReconstructor

grammar = r"""
start: pair+
pair: WORD "=" value ";"?
value: WORD | NUMBER
%import common.WS
%import common.WORD
%import common.NUMBER
%ignore WS
%ignore COMMENT
COMMENT: /#[^\n]*/
"""

config = """\
# server settings
host   =   localhost
port   =   8080     # change me

debug  =   yes
"""


def test():
    parser = Lark(grammar, parser='lalr', propagate_positions=True, maybe_placeholders=False)
    pr = PreservingReconstructor(parser)

    tree = pr.parse(config)

    # Change a single leaf value: port 8080 -> 9090
    port_pair = tree.children[1]
    port_pair.children[1].children[0] = Token('NUMBER', '9090')

    print(pr.reconstruct(tree))
    # Only the value changed; spacing, alignment and comments are intact.


if __name__ == '__main__':
    test()
