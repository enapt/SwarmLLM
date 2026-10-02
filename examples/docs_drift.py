#!/usr/bin/env python3
"""Report where the documentation has drifted from the code.

Four checks, each one a kind of drift a review by eye kept missing (2026-10-02:
together they found ~60 stale statements in docs that two sweeps that day had
already read):

  names     every backticked identifier a doc cites (`claim_room`, `SplitModel`,
            `FOO_BAR`) exists somewhere in the code or a pinned dependency
  paths     a backticked `module::path::item` names a module path the item is
            really defined under (`network::x` when x moved to `config::network`)
  diag      each row of docs/DIAGNOSTICS.md's DIAG tables gives the level and
            the fields of the tracing call that emits it (the repo_consistency
            guard checks only that the message exists)
  sections  every `<file>.md § "Heading"` pointer names a heading that exists

It is a REPORT, not a gate: some names are legitimately absent — a proposal in
docs/FUTURE_WORK.md, another project's function, a GGUF key the loader does
not read, a removed thing a "what it replaced" paragraph names on purpose.
Read each line before changing anything; log a deliberate non-change as
`wontfix` in .claude/sweep-log.jsonl so the next sweep skips it.

    python3 examples/docs_drift.py                 # all four checks, repo docs
    python3 examples/docs_drift.py diag sections   # just these
    python3 examples/docs_drift.py --memory        # also the auto-memory files
"""
import argparse
import os
import re
import subprocess
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEMORY = os.path.expanduser('~/.claude/projects/-home-user-SwarmLLM/memory/')
# Memory files a session actually loads or works from; round logs are history.
MEMORY_FILES = ['MEMORY.md', 'next_up.md', 'open_cautions.md', 'release_gate.md', 'testing_techniques.md',
                'code-map.md', 'code-map-backend.md', 'code-map-frontend.md', 'code-map-api.md']
CODE_EXT = ('.rs', '.js', '.py', '.sh', '.toml', '.yml', '.yaml', '.json', '.html', '.css', '.cu', '.cuh',
            '.c', '.h', '.ps1', '.service')
# Pinned dependencies a doc may legitimately name the API of.
DEP_PREFIXES = ('libp2p', 'quinn', 'candle', 'minijinja', 'tokio', 'axum', 'serde', 'redb', 'dashmap',
                'cudarc', 'reqwest')
IDENT = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')
TICK = re.compile(r'`([^`\n]{2,200})`')


def git_files():
    out = subprocess.run(['git', 'ls-files'], cwd=ROOT, capture_output=True, text=True).stdout
    return [f for f in out.split('\n') if f]


def read(path):
    try:
        with open(path if os.path.isabs(path) else os.path.join(ROOT, path), errors='ignore') as fh:
            return fh.read()
    except (FileNotFoundError, IsADirectoryError):
        return ''


def doc_files(files, with_memory):
    docs = [f for f in files if f.endswith('.md')
            and not f.startswith(('vendor/', 'docs/plans/archive/', 'docs/plans/benchmarks/'))
            and f not in ('docs/FUTURE_WORK_ARCHIVE.md', 'CHANGELOG.md')]
    if with_memory:
        docs += [MEMORY + f for f in MEMORY_FILES if os.path.exists(MEMORY + f)]
    return docs


def show(path):
    return path.replace(MEMORY, 'memory/')


def backticked(lines):
    """(line number, span) for every backticked span outside fenced blocks."""
    fenced = False
    for n, line in enumerate(lines, 1):
        if line.lstrip().startswith('```'):
            fenced = not fenced
            continue
        if not fenced:
            for span in TICK.findall(line):
                yield n, span.strip()


# ---------------------------------------------------------------- names

def check_names(files, docs):
    known = set()
    for f in files:
        if f.startswith('docs/') or f.endswith('.md'):
            continue
        if f.endswith(CODE_EXT) or f.startswith(('packaging/', 'config/', '.github/')):
            known.update(IDENT.findall(read(f)))
    registry = os.path.expanduser('~/.cargo/registry/src')
    if os.path.isdir(registry):
        for index in os.listdir(registry):
            base = os.path.join(registry, index)
            for crate in os.listdir(base):
                if crate.startswith(DEP_PREFIXES):
                    for dirpath, _, names in os.walk(os.path.join(base, crate)):
                        for name in names:
                            if name.endswith('.rs'):
                                known.update(IDENT.findall(read(os.path.join(dirpath, name))))

    def codelike(tok):
        return ('_' in tok.strip('_') and len(tok) >= 5) or \
            bool(re.fullmatch(r'(?:[A-Z][a-z0-9]+){2,}[A-Za-z0-9]*', tok))

    found = []
    for d in docs:
        seen = set()
        for n, span in backticked(read(d).split('\n')):
            if span.startswith(('--', 'SWARMLLM_', 'http', '/', '~', '$', '-')) or '/' in span:
                continue
            if ' ' in span and '::' not in span:
                continue
            if re.search(r'\.(rs|md|js|py|sh|toml|json|yml|cu|bin|gguf|log|html|css)\b', span):
                continue
            for tok in re.findall(r'[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)*', span):
                parts = tok.split('::')
                for p in parts:
                    if not (codelike(p) or (len(parts) > 1 and p == parts[-1])):
                        continue
                    if p in known or p.lower() in ('self', 'crate', 'super') or p in seen:
                        continue
                    seen.add(p)
                    found.append(f'{show(d)}:{n}\t{p}\t`{span[:90]}`')
    return found


# ---------------------------------------------------------------- paths

OUR_MODULES = {'api', 'inference', 'network', 'daemon', 'config', 'model', 'pool', 'crypto', 'credit', 'storage',
               'health', 'update', 'update_restart', 'update_signature', 'error', 'types', 'cli', 'identity',
               'swarmllm_types', 'http', 'split', 'pipeline', 'scheduler', 'router', 'dispatch', 'manager',
               'auto_manage', 'huggingface', 'layers', 'state', 'protocol'}


def check_paths(files, docs):
    defs = defaultdict(set)          # item name -> {(module path tuple, file)}
    texts = {}
    definer = re.compile(r'\b(?:fn|struct|enum|const|static|trait|type|mod|macro_rules!)\s+([A-Za-z_]\w*)')
    for f in files:
        if not (f.endswith('.rs') and f.startswith(('src/', 'crates/'))):
            continue
        parts = f[:-3].split('/')
        parts = [parts[1].replace('-', '_')] + parts[3:] if parts[0] == 'crates' else parts[1:]
        mod = tuple(p for p in parts if p not in ('mod', 'lib', 'main'))
        text = read(f)
        texts[f] = text
        for m in definer.finditer(text):
            defs[m.group(1)].add((mod, f))
        for body in re.finditer(r'enum\s+[A-Z]\w*[^{]*\{(.*?)\n\}', text, re.S):
            for v in re.finditer(r'^\s+([A-Z][A-Za-z0-9_]*)\b', body.group(1), re.M):
                defs[v.group(1)].add((mod, f))
        for body in re.finditer(r'struct\s+[A-Z]\w*[^{;]*\{(.*?)\n\}', text, re.S):
            for fld in re.finditer(r'^\s+(?:pub(?:\([^)]*\))?\s+)?([a-z_][a-z0-9_]*)\s*:', body.group(1), re.M):
                defs[fld.group(1)].add((mod, f))

    def file_has_type(f, ty):
        t = texts[f]
        return re.search(r'impl(?:<[^>]*>)?\s+(?:[\w:<>]+\s+for\s+)?(?:[\w:]+::)?' + ty + r'\b', t) or \
            re.search(r'(?:struct|enum|trait|union)\s+' + ty + r'\b', t)

    path_re = re.compile(r'\b((?:[A-Za-z_]\w*::)+[A-Za-z_]\w*)')
    found, seen = [], set()
    for d in docs:
        for n, span in backticked(read(d).split('\n')):
            for path in path_re.findall(span):
                segs = [s for s in path.split('::') if s not in ('crate', 'self', 'super', 'std', 'Self')]
                if len(segs) < 2 or segs[-1] not in defs:
                    continue
                name, mods = segs[-1], segs[:-1]
                if mods[0] not in OUR_MODULES and not mods[0][0].isupper():
                    continue
                ty = mods[-1] if mods[-1][0].isupper() else None
                wanted = [m for m in mods if not m[0].isupper()]
                ok = False
                for mod, f in defs[name]:
                    it = iter(mod)
                    if all(any(m == x for x in it) for m in wanted) and (not ty or file_has_type(f, ty)):
                        ok = True
                        break
                key = (d, path)
                if not ok and key not in seen:
                    seen.add(key)
                    where = sorted('::'.join(m) for m, _ in defs[name])[:3]
                    found.append(f'{show(d)}:{n}\t{path}\tdefined under {where}')
    return found


# ---------------------------------------------------------------- diag

def check_diag(files):
    calls = []   # (file, line, LEVEL, message, [fields])
    macro = re.compile(r'(?:tracing::)?(trace|debug|info|warn|error)!\s*\(')
    for f in files:
        if not (f.endswith('.rs') and f.startswith(('src/', 'crates/'))):
            continue
        t = read(f)
        for m in macro.finditer(t):
            i = j = m.end()
            depth = 1
            while j < len(t) and depth:
                c = t[j]
                if c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                elif c == '"':
                    j += 1
                    while j < len(t) and t[j] != '"':
                        j += 2 if t[j] == '\\' else 1
                j += 1
            body = t[i:j - 1]
            msg = re.search(r'"(DIAG:[^"]*)"', body)
            if not msg:
                continue
            head = re.sub(r'\([^()]*\)', '()', re.sub(r'"[^"]*"', '""', body[:msg.start()]))
            fields = []
            for part in head.split(','):
                part = part.strip()
                if not part or part.startswith(('target:', 'parent')):
                    continue
                name = part.split('=')[0].strip().lstrip('%?') if '=' in part else \
                    re.match(r'[%?]?([\w.]*)', part).group(1).split('.')[-1]
                if name and name not in fields:
                    fields.append(name)
            calls.append((f, t[:m.start()].count('\n') + 1, m.group(1).upper(), msg.group(1), fields))

    row = re.compile(r'^\|\s*(TRACE|DEBUG|INFO|WARN|ERROR)\s*\|\s*`(DIAG:[^`]*)`[^|]*\|\s*(.*?)\s*\|\s*$')
    found = []
    for n, line in enumerate(read('docs/DIAGNOSTICS.md').split('\n'), 1):
        m = row.match(line)
        if not m:
            continue
        level, msg = m.group(1), m.group(2).rstrip(' …')
        hits = [c for c in calls if c[3].startswith(msg) or msg.startswith(c[3])]
        if not hits:
            continue      # the repo_consistency guard owns "the message exists"
        exact = [c for c in hits if c[3] == msg] or hits
        problems = []
        levels = {c[2] for c in exact}
        if level not in levels:
            problems.append(f'level: doc {level}, code {"/".join(sorted(levels))}')
        doc_fields = set(re.findall(r'`([A-Za-z_]\w*)`', m.group(3)))
        code_fields = set().union(*[set(c[4]) for c in hits])
        missing = sorted(doc_fields - code_fields)
        if missing and code_fields:
            problems.append(f'fields the call does not log: {missing} (it logs {exact[0][4]})')
        if problems:
            found.append(f'docs/DIAGNOSTICS.md:{n}\t{msg}\t' + '; '.join(problems) + f'\t@{exact[0][0]}:{exact[0][1]}')
    return found


# ---------------------------------------------------------------- sections

def check_sections(files, with_memory):
    sources = [f for f in files if f.endswith(('.rs', '.md', '.sh', '.py', '.js', '.toml', '.yml'))
               and not f.startswith('vendor/') and f not in ('docs/FUTURE_WORK_ARCHIVE.md', 'CHANGELOG.md')]
    if with_memory and os.path.isdir(MEMORY):
        sources += [MEMORY + f for f in os.listdir(MEMORY)
                    if f.endswith('.md') and not f.startswith(('round_log', 'gotchas'))]

    def resolve(name):
        name = name.strip('`')
        candidates = [name, 'docs/' + name, 'docs/invariants/' + name, 'docs/plans/' + name,
                      '.claude/rules/' + name, '.claude/' + name, MEMORY + os.path.basename(name)]
        for c in candidates:
            if os.path.isfile(c if os.path.isabs(c) else os.path.join(ROOT, c)):
                return c
        return None

    heading_cache = {}

    def headings(path):
        if path not in heading_cache:
            heading_cache[path] = [re.sub(r'^#+\s*', '', l.strip()).replace('**', '').replace('`', '').lower()
                                   for l in read(path).split('\n') if l.startswith('#')]
        return heading_cache[path]

    def continuation(line):
        return re.sub(r'^\s*(?://[/!]?|#|>|\*|-)?\s*', '', line)

    pointer = re.compile(r'([\w./-]+\.md)`?\s*§\s*\\?"')
    found = []
    for src in sources:
        lines = read(src).split('\n')
        for i, line in enumerate(lines):
            for m in pointer.finditer(line):
                rest = line[m.end():]
                if '"' not in rest.replace('\\"', '') and i + 1 < len(lines):
                    rest += ' ' + continuation(lines[i + 1])
                title = re.split(r'\\?"', rest)[0]
                title = re.sub(r'\s+', ' ', title.replace('\\', '')).replace('`', '').replace('**', '')
                title = title.rstrip('…').strip().lower()
                if len(title) < 4:
                    continue
                target = resolve(m.group(1))
                if not target:
                    found.append(f'{show(src)}:{i + 1}\tno such file\t{m.group(1)}')
                    continue
                key = title[:45]
                if not any(h.startswith(key) or key in h for h in headings(target)):
                    found.append(f'{show(src)}:{i + 1}\tno such heading\t{m.group(1)} § "{title[:70]}"')
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('checks', nargs='*', choices=['names', 'paths', 'diag', 'sections', []], default=[])
    ap.add_argument('--memory', action='store_true', help='also check the auto-memory files')
    args = ap.parse_args()
    wanted = args.checks or ['names', 'paths', 'diag', 'sections']
    files = git_files()
    docs = doc_files(files, args.memory)
    total = 0
    for check in wanted:
        if check == 'names':
            found = check_names(files, docs)
        elif check == 'paths':
            found = check_paths(files, docs)
        elif check == 'diag':
            found = check_diag(files)
        else:
            found = check_sections(files, args.memory)
        print(f'== {check}: {len(found)}')
        for line in found:
            print(line)
        total += len(found)
    print(f'== {total} to read (a report, not a verdict — see the module docstring)', file=sys.stderr)


if __name__ == '__main__':
    main()
