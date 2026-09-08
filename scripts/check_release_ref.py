"""Reject publishing unless the exact version tag has passed CI on main."""
import json
import os
import subprocess
import tomllib
from pathlib import Path


def main():
    version = tomllib.loads(Path('Cargo.toml').read_text())['workspace']['package']['version']
    if os.environ.get('GITHUB_REF') != 'refs/tags/v' + version:
        raise SystemExit('Publishing requires a v<package-version> tag')
    subprocess.run(['git', 'merge-base', '--is-ancestor', 'HEAD', 'origin/main'], check=True)
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    repository = os.environ['GITHUB_REPOSITORY']
    checks = json.loads(subprocess.check_output([
        'gh', 'api', f'repos/{repository}/commits/{sha}/check-runs?per_page=100',
    ], text=True))['check_runs']
    # A rerun supersedes earlier attempts, including an earlier successful run.
    matching = [c for c in checks if c['name'] == 'CI' and c['app']['slug'] == 'github-actions']
    latest = max(matching, key=lambda c: c['id'], default=None)
    if latest is None or latest['conclusion'] != 'success':
        raise SystemExit('The exact release commit must have a successful Library CI check')
    print(f'Publication gate passed: v{version}, {sha}')


if __name__ == '__main__':
    main()
