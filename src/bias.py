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

import numpy as np
import pandas as pd

BIBBIAS_CACHE_PATH = Path(
    os.getenv("BIBBIAS_CACHE_PATH", str(Path.home() / ".cache" / "bibbias"))
)
BIBBIAS_CACHE_PATH.mkdir(exist_ok=True, parents=True)
GENDER_API_QUERY = "https://gender-api.com/get?name={name}&key={api_key}"


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

    # gender-api key
    api_key = os.getenv("GENDER_API_KEY", None)

    cached = None
    if (BIBBIAS_CACHE_PATH / "names.cache").exists():
        cached = json.loads((BIBBIAS_CACHE_PATH / "names.cache").read_text())

    # first pass
    resolved_df, cached = extract_metadata(bibstr, cached=cached, api_key=api_key)

    # Update cache
    (BIBBIAS_CACHE_PATH / "names.cache").write_text(json.dumps(dict(sorted(cached.items())), indent=2))

    resolved_df.to_csv('bibbias_output.tsv', index=False, sep='\t', na_rep='n/a')

    print(report_gender(resolved_df))

def extract_metadata(bibstr, cached=None, api_key=None):
    """Find the gender for a given bib file."""

    matches = re.findall(r"author\s=\s+\{(.*?)\}", bibstr, re.DOTALL)
    author_lists = [m.replace("\n", " ") for m in matches]
    bib_id = re.findall(r"@\w+\{(.*?),", bibstr)
    strip_initial = re.compile(r"\s*\w\.\s*")

    cached = cached or {}
    bib_entries = []
    dropped = {}
    for bid, authors in zip(bib_id, author_lists):
        data = {"bib_id": bid}

        if authors.startswith("{") and "}" not in authors:
            data["first_gender"] = "C"
            data["first_name"] = authors[1:]
            data["last_gender"] = np.nan
            data["last_name"] = np.nan
            bib_entries.append(data)
            continue

        authlst = authors.split(" and")
        first = strip_initial.sub("", authlst[0].strip().split(",")[-1].strip().lower())
        last = strip_initial.sub("", authlst[-1].strip().split(",")[-1].strip().lower())

        if first and last:
            first_gender = cached.get(first, None) or query_gender_api(first, api_key)
            if first_gender is not None:
                cached[first] = first_gender

            last_gender = cached.get(last, None) or query_gender_api(last, api_key)
            if last_gender is not None:
                cached[last] = last_gender

            data.update({
                "first_name": first,
                "first_gender": first_gender,
                "last_name": last,
                "last_gender":  last_gender,
            })
            bib_entries.append(data)
            continue

        dropped[bid] = authors

    return pd.DataFrame(bib_entries), cached


def query_gender_api(name, api_key):
    if api_key is None:
        return None

    print(f"Querying GenderAPI for name: {name}", file=sys.stderr)
    q = requests.get(GENDER_API_QUERY.format(name=name, api_key=api_key))
    if q.ok:
        response = q.json()
        accuracy = int(response["accuracy"])
        gender = response["gender"][0].upper() if accuracy >= 60 else "N"

        if accuracy < 40:
            gender = "U"
        
        return gender

    return None


def report_gender(df):
    """Generate a dictionary reporting gender of first and last authors."""

    total_n = df.shape[0]
    consortia_n = (df["first_gender"] == "C").sum()
    total = total_n - consortia_n
    first_male_n = (df["first_gender"] == "M").sum()
    first_female_n = (df["first_gender"] == "F").sum()
    last_male_n = (df["last_gender"] == "M").sum()
    last_female_n = (df["last_gender"] == "F").sum()
    first_neutral_n = (df["first_gender"] == "N").sum()
    last_neutral_n = (df["last_gender"] == "N").sum()

    female_first_and_last = ((df["first_gender"] == "F") & (df["last_gender"] == "F")).sum()
    female_first_male_last = ((df["first_gender"] == "F") & (df["last_gender"] == "M")).sum()
    male_first_female_last = ((df["first_gender"] == "M") & (df["last_gender"] == "F")).sum()

    return f"""Summary:
  - Total references = {total_n}.
  - Consortium: {consortia_n}.
  - Neutral/ambiguous gender: {first_neutral_n}/{last_neutral_n} (first/last).
  - Female first author: {first_female_n} ({100. * first_female_n / total:.02f}%).
  - Female last author: {last_female_n} ({100. * last_female_n / total:.02f}%).
  - Female first&last: {female_first_and_last} ({100. * female_first_and_last / total:.02f}%; {100. * female_first_and_last / first_female_n:.02f}% of female-first).
  - Female first, male last: {female_first_male_last} ({100. * female_first_male_last / total:0.2f}%; {100. * female_first_male_last / first_female_n:0.2f}% of female-first).
  - Male first author: {first_male_n} ({100. * first_male_n / total:.02f}%).
  - Male last author: {last_male_n} ({100. * last_male_n / total:.02f}%).
  - Male first, female last: {male_first_female_last} ({100. * male_first_female_last / total:0.2f}%; {100. * male_first_female_last / last_female_n:0.2f}% of female-last).
"""


if __name__ == "__main__":
    main()
