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
import sys
from pathlib import Path
import pandas as pd

from ruamel.yaml import YAML

# Read bib file
# to minimize bibtex -o minimized.bib texfile.aux
bibstr = Path("minimized.bib").read_text()
matches = re.findall(r'author\s=\s+\{(.*?)\}', bibstr, re.DOTALL)
author_lists = [m.replace("\n", " ") for m in matches]
bib_id = re.findall(r'@\w+\{(.*?),', bibstr)

gender_cache = json.loads(Path("names.cache").read_text())

# gender-api key
gender_api_query = "https://gender-api.com/get?name={name}&key={api_key}".format

data = {}
names = set()
for bid, authors in zip(bib_id, author_lists):
    authlst = authors.split(" and")
    first = strip_initial.sub("", authlst[0].strip().split(",")[-1].strip().lower())
    last = strip_initial.sub("", authlst[-1].strip().split(",")[-1].strip().lower())
    data[bid] = (
        (first, gender_cache.get(first, None)),
        (last, gender_cache.get(last, None)),
    )
    
    for f, n in data[bid]:
        if n is None:
            names.add(f)
            
for n, g in gender_cache.items():
    if g is None:
        print(f"Querying for {n}")
        q = requests.get(gender_api_query(name=n, api_key=api_key))
        if q.ok:
            responses[n] = q.json()
            if int(responses[n]["accuracy"]) >= 60:
                gender_cache[n] = "F" if responses[n]["gender"] == "female" else "M"

# Store cache

# Store responses?

nfirsts = {"M": 0, "F": 0}
nlasts = {"M": 0, "F": 0}
comb = {"MM": 0, "MF": 0, "FM": 0, "FF": 0}
for first, last in data.values():
    nfirsts[first[1]] += 1
    nlasts[last[1]] += 1
    comb[f"{first[1]}{last[1]}"] += 1
