"""Check tracked/staged publication files without printing secret values."""
import argparse
import os
import re
import subprocess
from pathlib import Path


def check(root, secret_file=None):
    names = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode('utf-8').split('\0')
    secrets = [v.encode() for k, v in os.environ.items() if k.endswith('_API_KEY') and v]
    if secret_file:
        for line in Path(secret_file).read_text(encoding='utf-8-sig').splitlines():
            if '=' in line and line.split('=', 1)[0].strip().endswith('_API_KEY'):
                value = line.split('=', 1)[1].strip().strip('\"\'')
                if value:
                    secrets.append(value.encode())
    problems, size = [], 0
    excluded = {'input', 'output', 'data', 'runs', 'tools', 'local-workspaces', '.venv', '.venv-review', 'dist', 'build'}
    patterns = [rb'gh[pousr]_[A-Za-z0-9]{30,}', rb'github_pat_[A-Za-z0-9_]{30,}',
                rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----']
    for name in filter(None, names):
        path = Path(name)
        if (path.parts[0] in excluded or path.suffix.lower() in {'.pdf', '.epub', '.png', '.jpg', '.pyc'}
                or path.name.startswith('.env') and path.name not in {'.env.example', '.env.local.example'}):
            problems.append((name, 'Private data, environment or binary path'))
        content = subprocess.check_output(['git', 'show', ':'+name], cwd=root)
        size += len(content)
        if len(content) > 1024*1024:
            problems.append((name, 'Unexpectedly large source file'))
        if any(re.search(pattern, content) for pattern in patterns) or any(secret in content for secret in secrets):
            problems.append((name, 'Possible credential; value omitted'))
    if problems:
        for name, reason in problems:
            print(f'{name}: {reason}')
        raise SystemExit(1)
    print(f'Publication check passed: {sum(bool(n) for n in names)} files, {size} bytes; no excluded data or detected credentials')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--secrets-from', help='Optional local .env file; checked only in memory')
    args = parser.parse_args()
    check(Path.cwd(), args.secrets_from)
