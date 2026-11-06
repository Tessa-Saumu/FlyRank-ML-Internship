"""Publish the capstone paper to the gh-pages branch.

The paper's source is work/paper/index.html - a single self-contained file
(its figures are embedded), so "building the site" is just copying that one
file. This script publishes it, plus a .nojekyll marker, to the root of the
gh-pages branch and pushes. GitHub Pages then serves it at the repository's
site URL.

Everything the project builds stays under work/; gh-pages is published
output only - never edit files there by hand.

The script never touches your working tree or your current branch: it
builds the gh-pages commit with plain git plumbing commands and pushes it.

Run from the repo root:

    python work/scripts/deploy_paper.py --dry-run   # preview what happens
    python work/scripts/deploy_paper.py             # publish

One-time GitHub setup (after the first deploy): Settings -> Pages ->
Source "Deploy from a branch" -> Branch: gh-pages, Folder: / (root).
"""

import os
import subprocess
import sys
import tempfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SOURCE = os.path.join(REPO, "work", "paper", "index.html")


def git(*args, check=True, env_extra=None, stdin=None):
    """Run a git command in the repo; return decoded stdout."""
    env = dict(os.environ)
    env.update(env_extra or {})
    if isinstance(stdin, str):
        stdin = stdin.encode("utf-8")
    r = subprocess.run(["git", *args], cwd=REPO, capture_output=True, input=stdin, env=env)
    if check and r.returncode != 0:
        sys.exit("git " + " ".join(args) + " failed:\n"
                 + r.stderr.decode("utf-8", "replace").strip())
    return r.stdout.decode("utf-8", "replace")


def site_url():
    """Derive the GitHub Pages URL from the origin remote, if possible."""
    remote = git("remote", "get-url", "origin", check=False).strip()
    if not remote or "github.com" not in remote:
        return None
    path = remote.replace(":", "/").replace("\\", "/")
    if path.endswith(".git"):
        path = path[:-4]
    parts = [p for p in path.split("/") if p]
    if len(parts) < 2:
        return None
    owner, repo = parts[-2], parts[-1]
    return f"https://{owner.lower()}.github.io/{repo}/"


def main():
    dry = "--dry-run" in sys.argv[1:]
    if not os.path.exists(SOURCE):
        sys.exit("paper not found: " + os.path.relpath(SOURCE, REPO))
    with open(SOURCE, "rb") as f:
        html = f.read()

    # Public-safety gates - refuse to publish a page that would fail the checks.
    if b"flyrank.ai" not in html:
        sys.exit("refusing to deploy: the page carries no flyrank.ai data credit")
    n_figs = html.count(b"data:image/png;base64,")
    if n_figs != 5:
        sys.exit(f"refusing to deploy: expected 5 embedded figures, found {n_figs}")

    branch = git("rev-parse", "--abbrev-ref", "HEAD").strip()
    short = git("rev-parse", "--short", "HEAD").strip()
    tip = git("ls-remote", "origin", "refs/heads/gh-pages", check=False).strip()
    parent = tip.split()[0] if tip else None
    url = site_url()

    print(f"source : work/paper/index.html ({len(html):,} bytes, {n_figs} embedded figures, data credit ok)")
    print(f"from   : {branch}@{short}")
    print("target : gh-pages branch"
          + (f" (update of {parent[:10]})" if parent else " (new branch)"))
    if url:
        print(f"site   : {url}  (once Pages serves gh-pages / root)")
    if dry:
        print("dry run - nothing pushed.")
        return

    # Build the gh-pages tree with plumbing: no checkout, no worktree changes.
    fd, tmp_index = tempfile.mkstemp(prefix="ghpages-index-")
    os.close(fd)
    try:
        idx = {"GIT_INDEX_FILE": tmp_index}
        git("read-tree", "--empty", env_extra=idx)
        html_sha = git("hash-object", "-w", "--stdin", stdin=html).strip()
        nojekyll_sha = git("hash-object", "-w", "--stdin", stdin=b"").strip()
        git("update-index", "--add", "--cacheinfo", "100644", html_sha,
            "index.html", env_extra=idx)
        git("update-index", "--add", "--cacheinfo", "100644", nojekyll_sha,
            ".nojekyll", env_extra=idx)
        tree = git("write-tree", env_extra=idx).strip()
        if parent and git("rev-parse", parent + "^{tree}").strip() == tree:
            print("gh-pages is already up to date - nothing to push.")
            return
        parents = ["-p", parent] if parent else []
        commit = git("commit-tree", tree, *parents, "-m",
                     f"Deploy paper (from {branch}@{short})").strip()
    finally:
        os.remove(tmp_index)

    git("push", "origin", f"{commit}:refs/heads/gh-pages")
    git("update-ref", "refs/heads/gh-pages", commit)
    print(f"pushed {commit[:10]} to gh-pages.")
    if url:
        print(f"live at {url} once Pages serves the gh-pages branch (Settings -> Pages).")
    else:
        print("could not derive the site URL from the origin remote - check Settings -> Pages.")


if __name__ == "__main__":
    main()
