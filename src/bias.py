#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2022 Oscar Esteban <code@oscaresteban.es>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
import os
import sys
import argparse
import json
import requests
from pathlib import Path
from collections import defaultdict
import re

BIBBIAS_CACHE_PATH = Path(
    os.getenv("BIBBIAS_CACHE_PATH", str(Path.home() / ".cache" / "bibbias"))
)
BIBBIAS_CACHE_PATH.mkdir(exist_ok=True, parents=True)


def _parser():
    parser = argparse.ArgumentParser(
        description="Run gender analytics on an existing BibTeX file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("bib_file", type=Path, help="The input bibtex file")
    return parser


def main(argv=None):
    """Execute querying."""

    pargs = _parser().parse_args(argv)
    # Read bib file
    # to minimize bibtex -o minimized.bib texfile.aux
    bibstr = pargs.bib_file.read_text()

    # first pass
    resolved, missed = find_gender(bibstr)

    if missed:
        resolved, missed = find_gender(bibstr, query_names(missed))

    print(report_gender(resolved))


def find_gender(bibstr, cached=None):
    """Find the gender for a given bib file."""

    matches = re.findall(r"author\s=\s+\{(.*?)\}", bibstr, re.DOTALL)
    author_lists = [m.replace("\n", " ") for m in matches]
    bib_id = re.findall(r"@\w+\{(.*?),", bibstr)
    strip_initial = re.compile(r"\s*\w\.\s*")

    if cached is None and (BIBBIAS_CACHE_PATH / "names.cache").exists():
        cached = json.loads((BIBBIAS_CACHE_PATH / "names.cache").read_text())

    cached = cached or {}

    data = {}
    missed = set()
    for bid, authors in zip(bib_id, author_lists):
        if authors.startswith("{") and "}" not in authors:
            authors = authors[1:]
            data[bid] = ((authors, "C"), (authors, "C"))
        else:
            authlst = authors.split(" and")
            first = strip_initial.sub("", authlst[0].strip().split(",")[-1].strip().lower())
            last = strip_initial.sub("", authlst[-1].strip().split(",")[-1].strip().lower())

            if first and last:
                data[bid] = (
                    (first, cached.get(first, None)),
                    (last, cached.get(last, None)),
                )
            else:
                print(f"Discarding reference {bid}: ('{first}', 'last').")

        for f, n in data[bid]:
            if n is None:
                missed.add(f)

    return data, missed


def query_names(nameset):
    """Lookup names in local cache, if not found hit Gender API."""

    cached = (
        json.loads((BIBBIAS_CACHE_PATH / "names.cache").read_text())
        if (BIBBIAS_CACHE_PATH / "names.cache").exists()
        else {}
    )
    misses = sorted(set(nameset) - set(cached.keys()))

    if not misses:
        return cached

    # gender-api key
    api_key = os.getenv("GENDER_API_KEY", None)
    if api_key is None:
        print(
            f"No Gender API key - {len(misses)} names could not be mapped: {misses}."
        )
        return cached

    gender_api_query = f"https://gender-api.com/get?name={{name}}&key={api_key}".format

    responses = {}
    for n in misses:
        print(f"Querying for {n}")
        q = requests.get(gender_api_query(name=n))
        if q.ok:
            responses[n] = q.json()
            accuracy = int(responses[n]["accuracy"])
            if accuracy >= 60:
                cached[n] = "F" if responses[n]["gender"] == "female" else "M"   
                print(f"{n}: {cached[n]} ({accuracy}%).")
            elif accuracy >= 40:
                cached[n] = "N"
            else:
                cached[n] = "Unknown"

    # Store cache
    (BIBBIAS_CACHE_PATH / "names.cache").write_text(json.dumps(cached, indent=2))

    # Store responses?
    return cached


def report_gender(data):
    """Generate a dictionary reporting gender of first and last authors."""

    summary = defaultdict(int)
    for first, last in data.values():
        summary[f"{first[1]}{last[1]}"] += 1

    total = float(sum(summary.values()))

    retval = f"""Summary:
  - Total authors = {total}.
  - Consortium: {summary['CC']}.
  - Unknown gender: first {sum(v for k, v in summary.items() if k.startswith("U"))}, last {sum(v for k, v in summary.items() if k.endswith("U"))}.
  - Woman first author: {100. * sum(v for k, v in summary.items() if k.startswith("F")) / total:.02f}%.
  - Woman last author: {100. * sum(v for k, v in summary.items() if k.endswith("F")) / total:.02f}%.
  - Woman first, and last: {100. * summary["FF"] / total}% (i.e., {100. * summary["FF"] / float(summary["FM"] + summary["FF"])}% of female-first).
  - Male last: {100. * (summary["FM"] + summary["MM"]) / total}%.
  - Male first, woman last: {100. * summary["MF"] / total}% ({100. * summary["MF"] / float(summary["MF"] + summary["FF"])}% of female-last).
"""
    return retval


if __name__ == "__main__":
    main()
