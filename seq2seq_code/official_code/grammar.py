from pyparsing import Word, alphas as pp_alpha, nums as pp_nums
import pyparsing as pp
import re
pp.ParserElement.enablePackrat()

# regex rules
alpha_regexp = r"""
(?!((?:THUMB-)?(?:IX|POSS|SELF)))   # negative lookahead for blocked glosses
[A-Z]                               # must start with uppercase
(?:                                 # optional middle section
    (?:                             # non-capturing group for allowed connectors
        (?:[-/][A-Z])               # hyphen or slash must be followed by uppercase
      | (?:_[0-9])                  # underscore must be followed by digit
      | (?:\+(?:[A-Z#]|fs-))       # plus + (uppercase OR # OR the literal fs-)
      | [A-Z0-9]                    # regular letter/digit continuation
    )
)*                                  # repeatable
(?:\.)?                             # optional trailing period
"""

# conventions kept for parsing

cl_prefix = pp.one_of(["CL", "DCL", "LCL", "SCL", "BCL", "BPCL", "PCL", "ICL"])
fs_prefix = pp.Literal("fs-")
index_core_ix = pp.Literal("IX")
other_index_core = pp.one_of(["POSS", "SELF"])
hashtag = pp.Literal("#")
dash = pp.Literal("-")
contraction = pp.Literal("^")
period = pp.Literal(".")
alpha = pp.Word(pp_alpha, max=1)
num = pp.Word(pp_nums, max=1)
word = pp.Regex(alpha_regexp, flags=re.X)

# grammar rules

full_grammar = pp.OneOrMore(
    fs_prefix |               # fingerspelling fs
    word |
    cl_prefix |               # classifiers like CL, DCL, etc.
    index_core_ix |           # IX
    other_index_core |        # POSS, SELF
    hashtag |                 # #
    contraction |             # ^
    period |                  # .
    dash |
    num |
    alpha                     # fallback LAST
)