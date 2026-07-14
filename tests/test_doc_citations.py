#!/usr/bin/env python
# -*- coding: utf8 -*-
"""Doc-citation lint: keeps docs/**/*.md (+ README.rst, FORK_CHANGES.md) honest
against the Python source they cite.

Plain unittest, no third-party dependencies, and (like tests/test_io.py /
tests/test_inference_core.py) no numpy/onnxruntime either -- this module only
reads text files and counts lines, so it must run identically on the base
install and on a full [ai] install. It imports nothing from libs/, precisely
so it can never accidentally acquire a heavy dependency.

WHAT THIS CHECKS
----------------
The doc surface is every ``docs/**/*.md`` file plus ``README.rst`` and
``FORK_CHANGES.md``. Within that surface it looks for citations of the shape
``path/to/file.py:NNN`` or ``path/to/file.py:NNN-MMM`` (bare filename or a
relative path), including the variants actually used in this repo:

* fully-qualified, backtick- or paren-wrapped: `` `libs/assist/controller.py:454-473` ``,
  ``(labelImg.py:1040)``.
* bare basename: `` `controller.py:79` `` (resolved by searching the repo tree
  for that basename -- see ``_resolve_path``).
* comma-separated line lists sharing one filename, e.g.
  `` `controller.py:57, 177-180` ``.
* "bare" citations that omit the filename and rely on context established
  earlier in the same document -- e.g. a ``## `labelImg.py` (2387줄)`` heading
  or a table's first column naming the file, followed by rows that only cite
  `` `:57` ``. The scanner tracks a running "current file" exactly the way a
  human reader would: top-to-bottom, updated every time a `` `something.py` ``
  token appears, and reused for every bare `` `:NNN` `` / ``(:NNN)`` that
  follows until the next file mention.

For every citation found, this test asserts:

1. the cited file exists (resolved relative to the repo root; a bare
   basename like ``controller.py`` is resolved by globbing the repo tree --
   see ``_resolve_path``);
2. every line in the cited (possibly single-line) range is within the file's
   actual current line count -- i.e. the citation does not point past EOF;
3. when a Python symbol (a ``def NAME``, ``class NAME``, or an ``ALL_CAPS``
   module-level constant) can be CONFIDENTLY tied to the citation, that name
   is actually defined/reachable at the cited location -- see the two
   heuristics below.

SYMBOL-EXTRACTION HEURISTIC (deliberately conservative)
--------------------------------------------------------
A citation's symbol is extracted with two mechanisms, tried in order:

1. **Positional list pairing** -- `` `sym1`/`sym2`(`cite1`/`cite2`) `` or
   `` `sym1`/`sym2`(`cite1`, `cite2`) `` (2 or more symbols immediately
   followed by "(", the same count of "/"- or ","-separated citations
   inside): pairs them up by position. This is what lets
   `` `accept_all`/`reject_all`(`:454`/`:475`) `` and
   `` `letterbox_params()`/`inverse_letterbox()`(`file:167-179`, `:182-216`) ``
   check EACH symbol against its OWN citation instead of one symbol
   swallowing its neighbour's citation too.
2. **Tight immediate adjacency** -- for any citation not already paired
   above, look at the text immediately preceding it: if the nearest
   non-punctuation token is a backtick-quoted identifier (allowing only
   whitespace/``,``/``/``/``(``/backticks in between -- i.e. no other prose),
   that identifier is the candidate symbol. This is deliberately narrow: an
   earlier version grabbed "the nearest preceding backtick token" with no
   adjacency requirement, which misfired on ordinary prose such as (from
   settings.md) ``` `reset()` 이후 `path=None`이라 ... `save()`는 저장되지
   않는다(`settings.py:24,45`) ``` -- the nearest backtick token is
   `` `save()` ``, but the citation is not "save is defined at 24 or 45", it
   is a cross-method behavioural note. There is no tight adjacency between
   `` `save()` `` and the citation (a full clause sits in between), so this
   mechanism correctly does not extract a symbol there, and the citation
   falls back to the range-only check.

Both mechanisms accept a DOTTED backtick token, e.g.
`` `AssistController._is_current`(`libs/assist/controller.py:363-372`) `` --
only the trailing component (``_is_current``) is captured and used as the
candidate symbol; the leading ``AssistController.`` qualifier is matched but
discarded (a bare method name is normally unique enough within one file, and
the "defined anywhere in the file" gate below still guards against a
coincidental match on an unrelated word).

A candidate symbol is then only trusted if it is actually defined
(``def``/``class``/assignment) SOMEWHERE in the resolved file at all -- this
catches the other class of false positive, e.g.
`` `segment`/`embed`(기본 `NotImplementedError`, `:66-85`) ``: with only one
symbol before the "(", positional pairing does not apply, and tight adjacency
finds ``NotImplementedError``, which is never *defined* (only raised) in
``libs/inference/backend.py`` -- so it is recognised as an unreliable
extraction and silently downgraded to the range-only check, exactly as asked
for: "do not force a symbol match that isn't really there."

WHERE THE SYMBOL IS ALLOWED TO BE, ONCE TRUSTED
------------------------------------------------
Some citations name a symbol only as an ANCHOR for a specific line of
interest inside that symbol's body, not its ``def``/``class`` line -- e.g.
(from ml-assist-architecture.md) `` `save_coco_format`(`libs/labelFile.py:79`) ``:
line 79 is the ``convert_points_to_bnd_box`` call INSIDE ``save_coco_format``
(whose own ``def`` is at line 56), not the ``def`` line itself. So a citation
passes if EITHER: the symbol's ``def``/``class``/assignment line falls within
a small tolerance window of the cited range (the strict, primary check --
this is what catches real drift, e.g. a stale class citation pointing at
some unrelated line number), OR the cited range falls entirely inside that
symbol's own body span (from its ``def``/``class`` line up to the next
sibling construct at the same or shallower indentation) -- the looser,
fallback check, which only ever widens a PASS, never narrows a failure: a
citation to a genuinely different construct still fails because it will not
be a real def/class match nor fall inside this symbol's body span.

A third, weakest fallback covers citing an exception at its OWN raise site
rather than its class statement -- e.g. (from annotation-formats.md)
`` `COCOParseError`(`libs/coco_io.py:219-220`) ``: line 220 is
``raise COCOParseError(...)``, not the ``class COCOParseError`` line (which
is elsewhere in the file, so it fails both stricter checks on its own). This
fallback only requires that the name appear as a whole word literally within
the cited lines -- still restrictive enough to fail a citation that drifted
onto genuinely unrelated code, where the name would not appear at all.

MODULES.MD LINE-COUNT CLAIMS
-----------------------------
Separately, every ``(NNN줄)`` claim in docs/reference/modules.md is checked
against the file's real Python line count (``len(f.readlines())``, matching
how the fork itself defines "line count" elsewhere in that doc).

KNOWN LIMITATION: ``TestModulesLineCounts`` only compares a file's TOTAL
line count against its ``(NNN줄)`` claim. A "compensating" edit -- N lines
inserted above a cited symbol and N lines deleted below it, elsewhere in the
SAME file -- leaves the total unchanged, so this tripwire stays silent even
though every citation past the insertion point has now drifted by N lines.
This is a deliberate, disclosed gap, not an oversight: closing it properly
would mean either (a) a per-citation content/identity check tight enough to
catch an N-line drift without also false-failing on ordinary, harmless
diffs -- a decorator added above a ``def``, a multi-line signature reflowed,
a comment inserted above a cited symbol -- all of which are exactly the
things ``_symbol_defined_near``'s ``slack=2`` window and
``_symbol_body_contains``'s body-span fallback exist to tolerate, or (b)
git-history-aware drift detection, which this lint deliberately has none of
(it only ever compares docs against the CURRENT checked-out source, per
"WHAT THIS CHECKS" above). Either would need meaningfully more machinery
than a doc-citation lint should carry, and (a) more precisely trades this
gap for tighter thresholds that would newly false-fail on the common,
harmless edits above; the risk/benefit did not favor closing it here. In
practice this whole-file-count check is not this lint's only defence
against drift either: any citation with a confidently-extracted symbol (see
SYMBOL-EXTRACTION HEURISTIC above) is independently range/body-checked by
``TestDocCitationSymbolsMatch`` regardless of whether the file's total line
count moved at all, so a compensating edit only slips past BOTH checks for a
citation that is simultaneously symbol-less (range-only) and lucky enough to
still land in-bounds after drifting.

NON-VACUOUSNESS
----------------
As of this writing, this lint extracts and range-checks 789 citation groups
across the whole doc surface, 280 of which also get a real symbol-location
check (computed by walking the extractor over every doc file -- not a guess).
``test_citation_extraction_is_non_vacuous`` asserts a lower bound on the
789 figure and prints it when run with ``-v``; ``TestDocCitationSymbolsMatch
.test_symbol_is_reachable_at_its_cited_location`` separately asserts a lower
bound on the 280 figure and prints IT when run with ``-v`` (two different
test methods, each printing its own count -- not one method printing both).

This was verified empirically, twice, with different-sized mutations of
``libs/assist/controller.py`` (each reverted afterwards and diffed back to
byte-identical against a pre-edit copy):

* **A single throwaway line** near the top (shifting every later line down by
  one) reliably trips exactly one assertion:
  ``TestModulesLineCounts.test_line_count_claims_match_reality`` (the
  ``libs/assist/controller.py`` line count in modules.md goes stale by
  exactly 1, e.g. ``542`` claimed vs ``543`` actual). It does NOT trip
  ``test_symbol_is_reachable_at_its_cited_location``: that check's tolerance
  (``_symbol_defined_near``'s ``slack=2`` window, and ``_symbol_body_contains``
  falling back to a symbol's current -- already-shifted-by-one -- body span)
  absorbs a 1-line drift for essentially every symbol citation in this repo's
  docs. A single line is simply too small a perturbation for the symbol-level
  checks to notice; the line-count check is what actually catches it, and it
  does so reliably.
* **A 6-line insertion** in the same spot additionally trips
  ``test_symbol_is_reachable_at_its_cited_location`` outright, with 14 named
  failures, each showing the doc file:line, the raw citation text, the
  expected symbol, and the actual current content at the (now-wrong) cited
  location -- e.g. ``docs/reference/modules.md:108: citation '`:454`' claims
  `accept_all` is defined in/reachable from ... at 454, but it is not``.

Both experiments are genuine evidence of non-vacuousness (the suite is not
merely "always green" against small tampering), but they catch drift through
different assertions at different magnitudes: line-count claims are exact and
therefore maximally sensitive (any drift at all is wrong by definition),
while the symbol/range checks carry deliberate slack for legitimate citation
styles (see above) and so need a larger, multi-line drift to trip reliably.
"""

import glob
import os
import re
import shutil
import tempfile
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

DOC_GLOB_PATTERNS = ['docs/**/*.md', 'README.rst', 'FORK_CHANGES.md']

# A citation's cited file, e.g. "libs/assist/controller.py" or "controller.py".
FILE_TOKEN = r'[A-Za-z0-9_][A-Za-z0-9_./\\-]*\.py'
NUM_LIST = r'\d+(?:-\d+)?(?:\s*,\s*\d+(?:-\d+)?)*'
IDENT = r'[A-Za-z_][A-Za-z0-9_]*'

# A backtick-quoted symbol may be dotted, e.g. `` `AssistController._is_current` ``
# -- only the LAST component (the method/attribute name) is captured and used
# for verification; the class-name prefix is accepted but discarded (a bare
# method name is normally unique enough within one file, and the "defined
# anywhere in the file" gate in TestDocCitationSymbolsMatch still guards
# against a coincidental false match on some unrelated word).
DOTTED_TAIL = r'(?:%s\.)*(%s)' % (IDENT, IDENT)

# Same grammar as DOTTED_TAIL, but fully non-capturing -- for embedding
# INSIDE another regex's own numbered groups (e.g. SYMBOL_GROUP_RE's repeated
# `{2,}` block below) without shifting that regex's group indices. All three
# symbol regexes (SYMBOL_TOKEN_RE, SYMBOL_GROUP_RE, TIGHT_PRECEDING_RE) must
# agree on what a symbol token looks like -- an earlier version left
# SYMBOL_GROUP_RE on plain IDENT (no dotted-tail support), so a positional
# group containing a dotted token like `` `LabelFile.convert_points_to_bnd_box` ``
# mis-detected the group boundary and fell through to the tight-adjacency
# fallback, which could pair the citation with the WRONG neighbouring symbol.
DOTTED_TAIL_NC = r'(?:%s\.)*(?:%s)' % (IDENT, IDENT)

# "libs/assist/controller.py:454-473" or "controller.py:57, 177-180" -- with
# or without a colon+number list (a bare file mention with no citation still
# has to update the "current file" context for later bare citations). The
# trailing `(?![-,\w])` anchor rejects a match that stops SHORT of a full
# citation-shaped token -- e.g. without it, the malformed "file.py:789-"
# would backtrack and quietly succeed as just "file.py:789", silently
# discarding the trailing "-" with zero signal; and "file.py:123abc" would
# truncate-match as "file.py:123", silently MIS-ACCEPTING a token that was
# never a real citation at all (a "123abc" line-number is not this repo's
# citation grammar in any form) as if it correctly cited line 123. `\w`
# subsumes `\d` (already needed for "789-") and additionally rejects any
# following letter/underscore -- including non-ASCII word characters, e.g. a
# Korean postposition glued directly onto the number like "454-473에서"
# ('에' is `\w` under Python's default Unicode matching) -- while still
# allowing the real terminators a valid citation actually ends on (backtick,
# paren, comma, whitespace, end of line: none of those are `\w`). With the
# anchor, that optional nums group simply fails to engage at all, leaving
# the whole ":789-"/":123abc" tail unconsumed -- which
# TestMalformedCitationsAreNeverSilentlySkipped below (a check deliberately
# independent of this regex) then catches and fails loudly on, naming the
# file:line and offending text, per this module's governing principle: a
# citation-shaped token this lint cannot fully parse must never disappear
# silently NOR be silently mis-accepted as valid.
FULL_RE = re.compile(
    r'(?P<file>%s)(?:\s*:\s*(?P<nums>%s)(?![-,\w]))?' % (FILE_TOKEN, NUM_LIST))

# "`:454`" / "(:454)" / "`:454`/`:475`" -- a citation with no filename of its
# own, relying on context. Deliberately requires a backtick/paren delimiter
# immediately around the colon (the only form actually used in this repo),
# to keep this from matching incidental "12:30"-shaped text elsewhere. The
# same `(?![-,\w])` anchor as FULL_RE guards against a truncated number list
# masquerading as a complete one (belt-and-suspenders: the mandatory closing
# backtick/paren already rejects truncated forms like "`:789-`" or
# "`:123abc`" today via the `\s*[`)]` requirement immediately after, but the
# explicit anchor keeps this regex correct even if that requirement is ever
# loosened).
BARE_RE = re.compile(r'[`(]\s*:\s*(?P<nums>%s)(?![-,\w])\s*[`)]' % NUM_LIST)

# One backtick-quoted identifier (optionally dotted, see DOTTED_TAIL above),
# optionally with a trailing "()" marking it as a function/method (stripped --
# the () is not part of the Python name).
SYMBOL_TOKEN_RE = re.compile(r'`%s(?:\(\))?`' % DOTTED_TAIL)

# 2+ symbols followed (optionally after whitespace, e.g. "`A`/`B` (`:10`/`:30`)")
# by "(...)" with no nested parens -- candidate for positional list pairing
# against the citations inside. Uses DOTTED_TAIL_NC (see above) so a group
# containing dotted tokens like `` `LabelFile.convert_points_to_bnd_box` ``
# is recognised as a single symbol, matching SYMBOL_TOKEN_RE/TIGHT_PRECEDING_RE.
SYMBOL_GROUP_RE = re.compile(
    r'((?:`%s(?:\(\))?`(?:\s*/\s*)?){2,})\s*\(([^()]*)\)' % DOTTED_TAIL_NC)

# Tight adjacency: a single backtick-symbol (optionally dotted, e.g.
# `` `AssistController._is_current` `` -- only the trailing method/attribute
# name is captured, see DOTTED_TAIL) immediately before a citation, separated
# only by punctuation actually used in this repo's docs (no prose).
TIGHT_PRECEDING_RE = re.compile(r'`%s(?:\(\))?`[\s,/(`]*$' % DOTTED_TAIL)

# docs/reference/modules.md "(NNN줄)" line-count claims, always right after a
# `path.py` mention.
LINE_COUNT_RE = re.compile(r'`(?P<file>%s)`\s*\((?P<count>\d+)줄' % FILE_TOKEN)

SYMBOL_DEF_TEMPLATE = (
    r'(?:\bclass\s+%(name)s\b)'
    r'|(?:\bdef\s+%(name)s\b)'
    r'|(?:^[ \t]*%(name)s\s*(?::[^=\n]+)?=(?!=))'
)

_EXCLUDED_DIR_PARTS = ('.git', '__pycache__', 'build', 'dist')


def _doc_paths():
    paths = []
    for pattern in DOC_GLOB_PATTERNS:
        paths.extend(glob.glob(os.path.join(REPO_ROOT, pattern), recursive=True))
    return sorted(set(paths))


def _repo_relative(path):
    return os.path.relpath(path, REPO_ROOT).replace(os.sep, '/')


_PATH_CACHE = {}


def _resolve_path(cited):
    """Resolve a citation's file text (bare basename or relative path) to an
    absolute path in this repo, or None if it cannot be found."""
    cited = cited.strip().replace('\\', '/')
    if cited in _PATH_CACHE:
        return _PATH_CACHE[cited]

    direct = os.path.join(REPO_ROOT, cited)
    if os.path.isfile(direct):
        _PATH_CACHE[cited] = direct
        return direct

    resolved = None
    if '/' not in cited:
        # Bare basename: search the repo tree for it (excluding vcs/build
        # noise), the way a human would.
        matches = []
        for root, dirnames, filenames in os.walk(REPO_ROOT):
            dirnames[:] = [d for d in dirnames if d not in _EXCLUDED_DIR_PARTS]
            if cited in filenames:
                matches.append(os.path.join(root, cited))
        if len(matches) == 1:
            resolved = matches[0]

    _PATH_CACHE[cited] = resolved
    return resolved


_FILE_LINES_CACHE = {}


def _file_lines(path):
    if path not in _FILE_LINES_CACHE:
        with open(path, 'r', encoding='utf-8') as handle:
            _FILE_LINES_CACHE[path] = handle.readlines()
    return _FILE_LINES_CACHE[path]


_CONSTRUCT_CACHE = {}


def _constructs(path):
    """[(lineno, indent, name), ...] for every top-level-or-method
    `class NAME` / `def NAME` line in `path`, sorted by line number. Used to
    find a symbol's body span (from its own line to the next sibling
    construct at the same or shallower indentation)."""
    if path in _CONSTRUCT_CACHE:
        return _CONSTRUCT_CACHE[path]
    pattern = re.compile(r'^(?P<indent>[ \t]*)(?:class|def)\s+(?P<name>\w+)')
    result = []
    for lineno, line in enumerate(_file_lines(path), start=1):
        match = pattern.match(line)
        if match:
            result.append((lineno, len(match.group('indent')), match.group('name')))
    _CONSTRUCT_CACHE[path] = result
    return result


def _parse_num_list(text):
    """'57, 177-180' -> [(57, 57), (177, 180)]

    Raises ValueError (not `assert`, which `python -O` strips silently) on a
    reversed range like '200-100' -- a citation whose start line is after its
    end line is never a valid line span."""
    ranges = []
    for chunk in text.split(','):
        chunk = chunk.strip()
        if '-' in chunk:
            start, end = chunk.split('-', 1)
            start, end = int(start), int(end)
        else:
            start = end = int(chunk)
        if start > end:
            raise ValueError(
                'malformed citation range %r in %r: start (%d) is after end (%d) '
                '-- a reversed range is never a valid line span' % (chunk, text, start, end))
        ranges.append((start, end))
    return ranges


# --- Malformed-citation detector --------------------------------------------
#
# GOVERNING PRINCIPLE: this lint must never silently skip -- nor silently
# MIS-ACCEPT -- a citation-shaped token it cannot fully parse. FULL_RE/BARE_RE
# above are anchored with a trailing `(?![-,\w])` so they never MATCH a
# truncated citation like "file.py:789-", "`:789-`", or "file.py:123abc" (the
# last of which, without the anchor, would truncate-match "123" and silently
# MIS-ACCEPT a token that was never a real citation as if it validly pointed
# at line 123). But refusing to match just makes the malformed token
# disappear from the citation set with zero signal, which is the exact
# failure this detector exists to close: it scans independently for anything
# that LOOKS like an attempted citation and demands the text after the colon
# be either genuinely not a citation attempt at all (see below), or a
# complete, well-formed, start<=end NUM_LIST not glued onto more text.
#
# Everything that could plausibly be part of a (possibly malformed) NUM_LIST
# attempt: digits, hyphens, commas, and the whitespace NUM_LIST itself
# tolerates around commas. Deliberately excludes letters, so ordinary prose
# immediately after a "file.py:" colon -- e.g. this very module docstring's
# own citation-format examples, `` `file.py:line` `` / `` `file.py:NNN` `` --
# is not mistaken for a citation attempt: "line"/"NNN" contain zero
# characters from this class, so nothing is captured and the following
# letter tells the detector this is prose, not a truncated number list.
_ATTEMPT_CHARS = r'[\d,\s-]*'

NUM_LIST_FULLMATCH_RE = re.compile(r'^%s$' % NUM_LIST)

# "libs/labelFile.py:" (+ an attempted, possibly malformed/empty, number
# list) -- anchored on FILE_TOKEN, so it fires regardless of whether the
# citation is backtick/paren-wrapped.
FILE_COLON_ATTEMPT_RE = re.compile(
    r'(?P<file>%s)\s*:\s*(?P<attempt>%s)' % (FILE_TOKEN, _ATTEMPT_CHARS))

# "`:" or "(:" (+ an attempted number list) -- the bare-citation equivalent.
BARE_COLON_ATTEMPT_RE = re.compile(
    r'(?P<open>[`(])\s*:\s*(?P<attempt>%s)' % _ATTEMPT_CHARS)


def _is_word_char(ch):
    """True for anything FULL_RE/BARE_RE's `(?![-,\\w])` anchor also rejects
    as a citation terminator -- ASCII letters/digits/underscore AND any other
    Unicode "word" character (Python's `\\w` is Unicode-aware by default),
    which covers e.g. a Korean postposition glued directly onto a number
    like "454-473에서" ('에' is `\\w`). Never true for the real terminators a
    valid citation actually ends on: backtick, paren, comma, whitespace, or
    end of line/string."""
    return bool(ch) and (ch.isalnum() or ch == '_')


def _malformed_attempt_message(where, prefix_text, attempt, following, allow_empty):
    """None if `attempt` (the text right after a citation-shaped colon) is
    either a well-formed NUM_LIST or genuinely not a citation attempt at
    all; otherwise a failure string naming `where` and the offending raw
    text. `following` is the raw text immediately after `attempt` (used both
    to classify what comes next and to show it in a failure message).

    `allow_empty` controls whether a completely empty attempt (no digits at
    all) is flagged when immediately closed by a backtick/paren:
    - For a FILE_TOKEN-anchored colon (e.g. "libs/labelFile.py:`"), this is
      unambiguous -- a `.py` file mention immediately followed by an empty,
      delimiter-closed colon is always an abandoned/forgotten citation, so
      `allow_empty` is False (empty IS flagged) there.
    - For a bare "`:"/"(:"-anchored colon, the opening delimiter is
      indistinguishable from the CLOSING backtick of some earlier, unrelated
      token followed by ordinary prose punctuation -- e.g. (from
      docs/reference/modules.md) `` `:85-88`: `stub`/`yolo_onnx` ``, where
      the second colon is just "table: description" prose, not a citation.
      BARE_RE itself sidesteps this ambiguity by requiring a MANDATORY
      number list (never optional), so this detector matches that same
      convention and does not flag an empty bare attempt; `allow_empty` is
      True (empty is NOT flagged) there. A non-empty-but-malformed bare
      attempt (e.g. "`:789-`") is unambiguous and still always flagged.
    """
    stripped = attempt.strip()
    next_char = following[:1]

    if stripped == '':
        if allow_empty:
            return None
        if next_char in ('`', ')'):
            return ('%s: citation-shaped text %r has a colon with NO line '
                     'number before the closing delimiter -- an abandoned '
                     'or forgotten citation, not a valid range'
                     % (where, prefix_text + ':'))
        return None  # ordinary prose colon (e.g. "`file.py:line`") -- not a citation attempt

    if not NUM_LIST_FULLMATCH_RE.match(stripped):
        return ('%s: citation-shaped text %r looks like a truncated or '
                 'malformed line-number list (it does not fully match a '
                 'well-formed number/range list -- e.g. a trailing "-", a '
                 'trailing ",", or a dangling range)'
                 % (where, prefix_text + ':' + attempt))

    # The numeric prefix parses cleanly on its own, but a well-formed NUM_LIST
    # match glued directly onto more word-characters with no delimiter in
    # between (e.g. "123abc", or "454-473에서") is not a citation at all --
    # it is a truncated match on unrelated text. Silently accepting it would
    # be worse than a silent skip: it would MIS-ACCEPT a non-citation token
    # as if it validly pointed at a real line number. See the `(?![-,\w])`
    # anchor on FULL_RE/BARE_RE above -- this is the same check, applied to
    # this independent detector's own (deliberately looser) attempt capture.
    if _is_word_char(next_char):
        return ('%s: citation-shaped text %r is a well-formed-looking number '
                 'list glued directly onto more text with no delimiter in '
                 'between (%r follows) -- not a real citation; silently '
                 'accepting this would MIS-ACCEPT it as validly citing '
                 'line(s) %s' % (where, prefix_text + ':' + attempt,
                                 following, stripped))

    try:
        _parse_num_list(stripped)
    except ValueError as exc:
        return '%s: citation-shaped text %r is malformed: %s' % (
            where, prefix_text + ':' + attempt, exc)

    return None


def _malformed_citation_shapes_in_doc(doc_path):
    """Scan `doc_path` for anything citation-shaped that FULL_RE/BARE_RE
    could not fully parse into a well-formed citation, and return a list of
    failure strings (empty if the whole doc is clean). See the
    "Malformed-citation detector" comment block above for why this exists as
    a check deliberately independent from FULL_RE/BARE_RE themselves."""
    failures = []
    with open(doc_path, 'r', encoding='utf-8') as handle:
        doc_lines = handle.readlines()

    for lineno, line in enumerate(doc_lines, start=1):
        where = '%s:%d' % (_repo_relative(doc_path), lineno)
        for match in FILE_COLON_ATTEMPT_RE.finditer(line):
            attempt = match.group('attempt')
            end = match.end('attempt')
            following = line[end:end + 12]
            msg = _malformed_attempt_message(
                where, match.group('file'), attempt, following, allow_empty=False)
            if msg:
                failures.append(msg)
        for match in BARE_COLON_ATTEMPT_RE.finditer(line):
            attempt = match.group('attempt')
            end = match.end('attempt')
            following = line[end:end + 12]
            msg = _malformed_attempt_message(
                where, match.group('open'), attempt, following, allow_empty=True)
            if msg:
                failures.append(msg)

    return failures


def _fmt_range(start, end):
    return str(start) if start == end else '%d-%d' % (start, end)


def _symbol_defined_anywhere(lines, name):
    pattern = re.compile(SYMBOL_DEF_TEMPLATE % {'name': re.escape(name)}, re.M)
    return any(pattern.search(line) for line in lines)


def _symbol_defined_near(lines, name, start, end, slack=2):
    """Strict check: def/class/assignment of `name` within
    [start-slack, end+slack] (1-indexed, inclusive). A decorator line
    directly above a def/class, or a multi-line def signature running past
    `end`, both fall inside this window."""
    pattern = re.compile(SYMBOL_DEF_TEMPLATE % {'name': re.escape(name)}, re.M)
    lo = max(1, start - slack)
    hi = min(len(lines), end + slack)
    for lineno in range(lo, hi + 1):
        if pattern.search(lines[lineno - 1]):
            return True
    return False


def _symbol_used_at(lines, name, start, end):
    """Weakest fallback: does `name` appear as a whole word literally inside
    the cited range at all? Covers citing an exception at its OWN raise site
    -- e.g. (from annotation-formats.md) `` `COCOParseError`(`libs/coco_io.py:219-220`) ``,
    where line 220 is `raise COCOParseError(...)`, not the class statement
    (that is elsewhere in the file, so the strict/body-span checks above
    both miss it on their own). Still meaningfully restrictive: it fails
    outright if the name is not even textually present at the cited lines,
    which is what catches a citation drifting onto unrelated code."""
    word = re.compile(r'\b%s\b' % re.escape(name))
    return any(word.search(lines[lineno - 1]) for lineno in range(start, end + 1))


def _symbol_body_contains(path, lines, name, start, end):
    """Looser fallback: does `name`'s own body (from its def/class line to
    the next sibling construct at the same-or-shallower indentation) contain
    the whole cited range? Only ever widens a pass -- a citation to some
    other construct entirely still will not fall inside this span."""
    constructs = _constructs(path)
    candidates = [c for c in constructs if c[2] == name]
    for def_line, indent, _name in candidates:
        body_end = len(lines)
        for other_line, other_indent, _other_name in constructs:
            if other_line > def_line and other_indent <= indent:
                body_end = other_line - 1
                break
        if def_line <= start and end <= body_end:
            return True
    return False


class Citation(object):
    """One citation: a file, one-or-more (start, end) line ranges sharing a
    single raw match, and (maybe) a confidently-extracted symbol name."""

    def __init__(self, doc_path, doc_lineno, raw_text, file_text, ranges, symbol):
        self.doc_path = doc_path
        self.doc_lineno = doc_lineno
        self.raw_text = raw_text
        self.file_text = file_text
        self.ranges = ranges
        self.symbol = symbol

    def where(self):
        return '%s:%d' % (_repo_relative(self.doc_path), self.doc_lineno)


def _split_top_level(text, delimiter):
    return [piece.strip() for piece in text.split(delimiter)]


def _extract_citations_from_doc(doc_path):
    """Scan one doc file top-to-bottom, tracking a running "current file" for
    bare `:NNN` citations, and return a list of Citation objects."""
    citations = []
    current_file = None

    with open(doc_path, 'r', encoding='utf-8') as handle:
        doc_lines = handle.readlines()

    for lineno, line in enumerate(doc_lines, start=1):
        pending = []  # [{'file','ranges','start','end','symbol'}, ...] on this line
        claimed_spans = []

        # Both regexes' matches are walked together IN TEXT ORDER (not "all
        # FULL_RE matches, then all BARE_RE matches") -- a bare citation must
        # pick up whatever file was mentioned immediately before IT, not
        # whichever file happens to be mentioned last anywhere on the line.
        full_matches = [('full', m) for m in FULL_RE.finditer(line)]
        bare_matches = [('bare', m) for m in BARE_RE.finditer(line)]
        for kind, match in sorted(full_matches + bare_matches, key=lambda km: km[1].start()):
            if kind == 'full':
                current_file = match.group('file')
                nums = match.group('nums')
                if nums:
                    claimed_spans.append(match.span())
                    pending.append({'file': current_file, 'ranges': _parse_num_list(nums),
                                    'start': match.start(), 'end': match.end(), 'symbol': None})
            else:
                span = match.span()
                if any(a <= span[0] < b for a, b in claimed_spans):
                    continue  # already part of a full "file.py:NNN" match above
                if current_file is None:
                    continue  # no context established yet on this doc -- skip
                pending.append({'file': current_file, 'ranges': _parse_num_list(match.group('nums')),
                                'start': match.start(), 'end': match.end(), 'symbol': None})

        # Mechanism 1: positional list pairing, e.g.
        # "`accept_all`/`reject_all`(`:454`/`:475`)".
        for group_match in SYMBOL_GROUP_RE.finditer(line):
            symbols = SYMBOL_TOKEN_RE.findall(group_match.group(1))
            inner_start, inner_end = group_match.span(2)
            members = [c for c in pending if inner_start <= c['start'] and c['end'] <= inner_end]
            if len(members) != len(symbols):
                continue
            for delimiter in ('/', ','):
                pieces = _split_top_level(group_match.group(2), delimiter)
                if len(pieces) == len(symbols):
                    for cite, symbol in zip(members, symbols):
                        cite['symbol'] = symbol
                    break

        # Mechanism 2: tight immediate adjacency, for anything mechanism 1
        # did not confidently resolve.
        for cite in pending:
            if cite['symbol'] is not None:
                continue
            match = TIGHT_PRECEDING_RE.search(line[:cite['start']])
            if match:
                cite['symbol'] = match.group(1)

        for cite in pending:
            citations.append(Citation(
                doc_path, lineno, line[cite['start']:cite['end']],
                cite['file'], cite['ranges'], cite['symbol']))

    return citations


class TestDocCitationsExist(unittest.TestCase):
    """Every file.py:NNN[-MMM] citation must point at a real, in-bounds
    location in a file that actually exists."""

    @classmethod
    def setUpClass(cls):
        cls.all_citations = []
        for doc_path in _doc_paths():
            cls.all_citations.extend(_extract_citations_from_doc(doc_path))

    def test_cited_files_exist_and_ranges_are_in_bounds(self):
        failures = []
        for citation in self.all_citations:
            resolved = _resolve_path(citation.file_text)
            if resolved is None:
                failures.append(
                    '%s: citation %r names %r, which does not resolve to any '
                    'file in this repo (checked as a relative path and, if '
                    'bare, by searching the whole tree for that basename).'
                    % (citation.where(), citation.raw_text, citation.file_text))
                continue
            lines = _file_lines(resolved)
            total = len(lines)
            for start, end in citation.ranges:
                if start < 1 or end > total:
                    failures.append(
                        '%s: citation %r -> %s:%s is out of bounds -- %s has '
                        '%d lines. Actual content near the end of the file:\n'
                        '  ...%s'
                        % (citation.where(), citation.raw_text,
                          _repo_relative(resolved), _fmt_range(start, end),
                          _repo_relative(resolved), total,
                          ''.join(lines[max(0, total - 3):]).rstrip()))

        if failures:
            self.fail('%d stale citation(s):\n\n%s'
                      % (len(failures), '\n\n'.join(failures)))

    def test_citation_extraction_is_non_vacuous(self):
        # If this ever collapses to ~0, the regexes broke, not the docs.
        self.assertGreater(
            len(self.all_citations), 200,
            'the citation extractor only found %d citations across the whole '
            'doc surface -- that is suspiciously low; the extraction regexes '
            'themselves are probably broken (see the module docstring for '
            'the citation forms this is supposed to recognise)'
            % len(self.all_citations))
        print('\n[test_doc_citations] guarding %d citation group(s) across '
             '%d doc file(s)' % (len(self.all_citations), len(_doc_paths())))


class TestMalformedCitationsAreNeverSilentlySkipped(unittest.TestCase):
    """GOVERNING PRINCIPLE: if this lint finds a citation-shaped token it
    cannot fully parse, that must be a hard failure naming the file:line and
    offending text -- never a silent skip, and never a silent MIS-ACCEPT.
    FULL_RE/BARE_RE are anchored so a truncated citation like "file.py:789-",
    "`:789-`", or "file.py:123abc" never MATCHES, but a regex that just
    refuses to match a malformed token makes it vanish from the citation set
    with zero signal (or, for "123abc", would truncate-match and silently
    accept "123" as a real citation to line 123 if the anchor were weaker) --
    this test is the independent tripwire that catches both failure modes
    (see the "Malformed-citation detector" comment block above
    `_ATTEMPT_CHARS` for the full mechanism)."""

    def test_no_doc_has_an_unparseable_citation_shaped_token(self):
        failures = []
        for doc_path in _doc_paths():
            failures.extend(_malformed_citation_shapes_in_doc(doc_path))
        if failures:
            self.fail('%d malformed citation-shaped token(s) found -- a '
                      'citation this lint could not fully parse must never '
                      'disappear silently:\n\n%s'
                      % (len(failures), '\n\n'.join(failures)))


class TestMalformedCitationDetectorItself(unittest.TestCase):
    """Exercises `_malformed_citation_shapes_in_doc` -- the actual function
    the test above runs against the real doc surface -- directly, against
    controlled fixtures the real doc surface does not (and must not)
    contain. This is what proves the detector really catches what it claims
    to, not merely that the real docs happen to be clean."""

    def _check(self, snippet):
        # dir=REPO_ROOT: on Windows, os.path.relpath (used by _repo_relative,
        # which this exercises) raises ValueError across drive letters, and
        # the platform default tempdir is not guaranteed to share a drive
        # with the repo checkout.
        fixture_dir = tempfile.mkdtemp(prefix='doc_citation_lint_fixture_', dir=REPO_ROOT)
        try:
            fixture_path = os.path.join(fixture_dir, 'fixture.md')
            with open(fixture_path, 'w', encoding='utf-8') as handle:
                handle.write(snippet)
            return _malformed_citation_shapes_in_doc(fixture_path)
        finally:
            shutil.rmtree(fixture_dir)

    def test_trailing_hyphen_after_full_citation_is_caught(self):
        # FULL_RE backtracks "789-" down to just "789" and would otherwise
        # silently drop the trailing "-" with zero signal.
        failures = self._check('see `libs/labelFile.py:789-` for details\n')
        self.assertEqual(1, len(failures), failures)
        self.assertIn('789-', failures[0])

    def test_trailing_hyphen_after_bare_citation_is_caught(self):
        # A bare "`:789-`" fails BARE_RE entirely (no closing delimiter
        # immediately after the number list) and vanishes with zero signal.
        failures = self._check(
            '`libs/labelFile.py:100` and also `:789-` nearby\n')
        self.assertTrue(any('789-' in f for f in failures), failures)

    def test_missing_line_number_is_caught(self):
        failures = self._check('nothing here: `libs/labelFile.py:`\n')
        self.assertEqual(1, len(failures), failures)
        self.assertIn('labelFile.py:', failures[0])

    def test_reversed_range_is_caught(self):
        failures = self._check('`libs/labelFile.py:200-100`\n')
        self.assertEqual(1, len(failures), failures)
        self.assertIn('200', failures[0])
        self.assertIn('100', failures[0])

    def test_trailing_comma_is_caught(self):
        failures = self._check('`libs/labelFile.py:57,`\n')
        self.assertEqual(1, len(failures), failures)

    def test_number_glued_to_trailing_letters_is_caught(self):
        # Regression for a Codex review finding: without the `(?![-,\w])`
        # anchor (extended from `(?![-,\d])`), FULL_RE truncate-matches
        # "controller.py:123abc" as just "controller.py:123" and silently
        # MIS-ACCEPTS it as a valid citation to line 123 -- worse than a
        # silent skip, since the token never disappears, it gets accepted as
        # something it is not.
        failures = self._check('see `libs/controller.py:123abc` for details\n')
        self.assertEqual(1, len(failures), failures)
        self.assertIn('123', failures[0])

    def test_number_glued_to_korean_postposition_is_caught(self):
        # Same failure class as the ASCII case above, but with a Korean
        # postposition glued directly onto the number with no delimiter --
        # `\w` is Unicode-aware, so this must be caught too, not just ASCII
        # suffixes.
        failures = self._check('`libs/labelFile.py:454-473에서` 문제가 생긴다\n')
        self.assertEqual(1, len(failures), failures)
        self.assertIn('454-473', failures[0])

    def test_korean_postposition_after_a_closed_citation_is_not_flagged(self):
        # Contrast case: this repo's docs routinely follow a real,
        # backtick-closed citation with a Korean postposition, e.g.
        # "...`(labelImg.py:1058-1069)`가 호출자가...". The postposition
        # comes AFTER the closing delimiter, not glued onto the number
        # itself, so this must NOT be flagged.
        failures = self._check('`libs/labelFile.py:454-473`에서 문제가 생긴다\n')
        self.assertEqual([], failures)

    def test_well_formed_citation_is_not_flagged(self):
        failures = self._check('`libs/labelFile.py:182-205`\n')
        self.assertEqual([], failures)

    def test_well_formed_comma_list_is_not_flagged(self):
        failures = self._check('`controller.py:57, 177-180`\n')
        self.assertEqual([], failures)

    def test_prose_placeholder_is_not_flagged(self):
        # Mirrors this module's own docstring usage: "`file.py:line`" /
        # "`file.py:NNN`" describe the citation FORMAT, they are not
        # themselves citations, and must not be mistaken for malformed ones.
        failures = self._check(
            'cite as `file.py:line` or `file.py:NNN` in this repo\n')
        self.assertEqual([], failures)

    def test_term_colon_prose_after_a_real_citation_is_not_flagged(self):
        # Mirrors a real occurrence in docs/reference/modules.md: a closing
        # backtick of an unrelated earlier token immediately followed by
        # ordinary "term: description" prose punctuation must not be
        # mistaken for an abandoned bare citation.
        failures = self._check(
            '이름→백엔드 생성 테이블(`:85-88`: `stub`/`yolo_onnx`)\n')
        self.assertEqual([], failures)


class TestSymbolGroupRegexAgreesOnDottedNames(unittest.TestCase):
    """Regression test for SYMBOL_GROUP_RE: it must agree with
    SYMBOL_TOKEN_RE/TIGHT_PRECEDING_RE about what a symbol token looks like,
    including DOTTED tokens like `` `AssistController.accept_all` ``. An
    earlier version left SYMBOL_GROUP_RE on plain IDENT (no dotted-tail
    support), so a positional-list group containing a dotted token failed to
    match SYMBOL_GROUP_RE at all -- every symbol in that group then fell
    through to the tight-adjacency fallback (mechanism 2), which can
    associate the WRONG symbol with a citation when a group has more than
    one citation. This is exercised against a synthetic fixture, not the
    real doc surface: as of this writing, every dotted positional-list-
    shaped mention in the real docs (see docs/how-to/verify-and-difficult.md
    and docs/explanation/ml-assist-architecture.md) happens to share ONE
    citation across multiple symbols, which mechanism 1 correctly declines
    to pair (its symbol/citation counts must match) regardless of this fix
    -- so this fix currently changes zero real citations' pairing, but it
    would misfire on the very first doc that DOES cite two dotted symbols
    against two separate citations, which is exactly the shape this test
    fixture uses."""

    def _extract(self, snippet):
        fixture_dir = tempfile.mkdtemp(prefix='doc_citation_lint_fixture_', dir=REPO_ROOT)
        try:
            fixture_path = os.path.join(fixture_dir, 'fixture.md')
            with open(fixture_path, 'w', encoding='utf-8') as handle:
                handle.write(snippet)
            return _extract_citations_from_doc(fixture_path)
        finally:
            shutil.rmtree(fixture_dir)

    def test_dotted_symbols_pair_with_their_own_citation_not_their_neighbours(self):
        # File-path citations (not bare `:NNN`) deliberately, since the
        # positional split is "/"-delimited and a file path also contains
        # "/" -- using bare citations (as this repo's docs actually do for
        # this exact form, e.g. "`accept_all`/`reject_all`(`:454`/`:475`)")
        # sidesteps that unrelated ambiguity and isolates what this test is
        # actually about: whether SYMBOL_GROUP_RE recognises a DOTTED symbol
        # as one token of the "/"-separated list at all.
        citations = self._extract(
            '`libs/assist/controller.py:1` establishes context.\n'
            '`AssistController.accept_all`/`AssistController.reject_all`'
            '(`:454`/`:475`)\n')
        dotted = [c for c in citations if c.ranges[0][0] in (454, 475)]
        self.assertEqual(2, len(dotted), dotted)
        by_start = sorted(dotted, key=lambda c: c.ranges[0][0])
        self.assertEqual('accept_all', by_start[0].symbol)
        self.assertEqual([(454, 454)], by_start[0].ranges)
        self.assertEqual('reject_all', by_start[1].symbol)
        self.assertEqual([(475, 475)], by_start[1].ranges)


class TestDocCitationSymbolsMatch(unittest.TestCase):
    """Where a citation's symbol can be confidently extracted (see module
    docstring) and that name is defined SOMEWHERE in the file, assert it is
    reachable at the cited location (strict def/class window, or -- for
    citations that anchor a line of interest inside a body -- the symbol's
    own body span)."""

    @classmethod
    def setUpClass(cls):
        cls.all_citations = []
        for doc_path in _doc_paths():
            cls.all_citations.extend(_extract_citations_from_doc(doc_path))

    def test_symbol_is_reachable_at_its_cited_location(self):
        failures = []
        checked = 0
        for citation in self.all_citations:
            if not citation.symbol:
                continue
            resolved = _resolve_path(citation.file_text)
            if resolved is None:
                continue  # already reported by the other test class
            lines = _file_lines(resolved)
            if not _symbol_defined_anywhere(lines, citation.symbol):
                # Not confidently a definition target after all (e.g. a
                # builtin exception name, or a base class mentioned in
                # passing) -- fall back to the range-only check silently.
                continue

            checked += 1
            ok = any(
                _symbol_defined_near(lines, citation.symbol, start, end)
                or _symbol_body_contains(resolved, lines, citation.symbol, start, end)
                or _symbol_used_at(lines, citation.symbol, start, end)
                for start, end in citation.ranges)
            if not ok:
                actual = []
                for start, end in citation.ranges:
                    lo, hi = max(1, start - 2), min(len(lines), end + 2)
                    actual.append('%s:%s currently reads:\n    %s'
                                  % (_repo_relative(resolved), _fmt_range(start, end),
                                    '    '.join(lines[lo - 1:hi]).rstrip()))
                failures.append(
                    '%s: citation %r claims `%s` is defined in/reachable from '
                    '%s at %s, but it is not (it IS defined elsewhere in that '
                    'file, so this is a stale line number, not a bad '
                    'extraction). %s'
                    % (citation.where(), citation.raw_text, citation.symbol,
                      citation.file_text,
                      ' or '.join(_fmt_range(s, e) for s, e in citation.ranges),
                      ' / '.join(actual)))

        if failures:
            self.fail('%d symbol citation(s) point at the wrong line(s):\n\n%s'
                      % (len(failures), '\n\n'.join(failures)))
        print('\n[test_doc_citations] %d citation(s) had a confidently-extracted, '
             'file-verified symbol checked against their cited location' % checked)
        self.assertGreater(
            checked, 30,
            'only %d citations had a confidently-extracted, file-verified '
            'symbol to check -- the symbol-adjacency heuristics may have '
            'regressed (see the module docstring)' % checked)


class TestModulesLineCounts(unittest.TestCase):
    """Every `(NNN줄)` claim in docs/reference/modules.md must equal the
    file's real Python line count."""

    def test_line_count_claims_match_reality(self):
        modules_md = os.path.join(REPO_ROOT, 'docs', 'reference', 'modules.md')
        with open(modules_md, 'r', encoding='utf-8') as handle:
            doc_lines = handle.readlines()

        failures = []
        checked = 0
        for lineno, line in enumerate(doc_lines, start=1):
            for match in LINE_COUNT_RE.finditer(line):
                file_text = match.group('file')
                claimed = int(match.group('count'))
                resolved = _resolve_path(file_text)
                if resolved is None:
                    failures.append(
                        '%s:%d: %r does not resolve to a file in this repo'
                        % (_repo_relative(modules_md), lineno, file_text))
                    continue
                checked += 1
                actual = len(_file_lines(resolved))
                if actual != claimed:
                    failures.append(
                        '%s:%d claims `%s` is %d줄, but it is actually %d '
                        'lines (len(open(f).readlines()))'
                        % (_repo_relative(modules_md), lineno, file_text,
                          claimed, actual))

        if failures:
            self.fail('%d stale line-count claim(s):\n\n%s'
                      % (len(failures), '\n'.join(failures)))
        self.assertGreater(checked, 15,
                           'only %d "(NNN줄)" claims were found in modules.md -- '
                           'the LINE_COUNT_RE pattern may have stopped matching '
                           'the real markup' % checked)


if __name__ == '__main__':
    unittest.main()
