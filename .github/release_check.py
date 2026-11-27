#!/usr/bin/env python3
"""Refuse to publish unless the pushed tag is a release of the default branch at the version the package says.

    python3 .github/release_check.py --tag v1.4.1 --branch origin/main [--repo .]

The publish workflow runs this in a job of its own, before the job that holds the publishing identity. It fails when:
  * the tag is not v<version>, where <version> is pyproject.toml's [project] version;
  * server.json names another version anywhere;
  * the tagged commit is not on the default branch (a tag pushed on a side branch, or on a commit that edits the
    workflow without being merged, publishes nothing).
Standard library only.
"""
import argparse
import json
import os
import re
import subprocess
import sys


def pyproject_version(text):
    """The version in pyproject.toml's [project] table, or None."""
    section = None
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("["):
            section = s.strip("[] ")
            continue
        if section == "project":
            m = re.match(r'version\s*=\s*"([^"]+)"', s)
            if m:
                return m.group(1)
    return None


def server_versions(text):
    """Every "version" value in server.json, at any depth."""
    found = []

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "version" and isinstance(v, str):
                    found.append(v)
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    walk(json.loads(text))
    return found


def problems(tag, branch, repo="."):
    out = []
    try:
        with open(os.path.join(repo, "pyproject.toml"), encoding="utf-8") as fh:
            version = pyproject_version(fh.read())
    except OSError as exc:
        return ["pyproject.toml could not be read: %s" % exc]
    if not version:
        return ["pyproject.toml has no [project] version"]
    if tag != "v" + version:
        out.append("the tag is %r and pyproject.toml's version is %s, so the tag must be v%s" % (tag, version, version))
    server = os.path.join(repo, "server.json")
    if os.path.isfile(server):
        try:
            with open(server, encoding="utf-8") as fh:
                other = sorted(set(v for v in server_versions(fh.read()) if v != version))
        except ValueError as exc:
            other = ["(unreadable: %s)" % exc]
        if other:
            out.append("server.json names version(s) %s beside pyproject's %s" % (", ".join(other), version))
    r = subprocess.run(["git", "-C", repo, "rev-parse", "--verify", "--quiet", tag + "^{commit}"],
                       capture_output=True, text=True)
    commit = r.stdout.strip()
    if r.returncode != 0 or not commit:
        out.append("the tag %r does not name a commit in this clone" % tag)
        return out
    r = subprocess.run(["git", "-C", repo, "merge-base", "--is-ancestor", commit, branch], capture_output=True, text=True)
    if r.returncode == 1:
        out.append("the tagged commit %s is not on %s" % (commit[:12], branch))
    elif r.returncode != 0:
        out.append("git could not compare the tagged commit with %s: %s" % (branch, (r.stderr or "").strip()[:200]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--branch", required=True, help="the default branch as this clone names it, e.g. origin/main")
    ap.add_argument("--repo", default=".")
    a = ap.parse_args()
    found = problems(a.tag, a.branch, a.repo)
    for p in found:
        print("REFUSING TO PUBLISH - %s" % p)
    if not found:
        print("%s is a release of %s at the version the package says" % (a.tag, a.branch))
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
