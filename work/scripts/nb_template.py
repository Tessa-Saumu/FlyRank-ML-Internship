"""FlyRank original template notebook text (cells copied verbatim from the fork baseline b34f9da).

build_notebooks.py reads the title cell, the numbered section headings with their italic prompts and the
Self-check cell from here, so every notebook keeps the original structure. The Colab badge line is a placeholder."""
TEMPLATE = {'w01_research_question': {'title': '# ML-02 \u2014 Research Question and Provisional Lane\n'
                                    '\n'
                                    '{BADGE}\n'
                                    '\n'
                                    'This skeleton is yours to fill. Work the sections **in order** \u2014 each one has a '
                                    'one-line hint. Simple words, honest numbers.\n'
                                    '\n'
                                    '> Working with an AI assistant? Tell it to read `skills/README.md` first and '
                                    'load the one skill this assignment names on its card.',
                           'sec': ['## 1. My lane (or freestyle) and why\n'
                                   '\n'
                                   "*Name your lane \u2014 or say 'freestyle' and describe your own question. One short "
                                   'paragraph: why this one?*',
                                   '## 2. The question: decision, action, cost of a wrong call\n'
                                   '\n'
                                   '*What decision does your work improve? Who acts on it? What does a wrong '
                                   'recommendation cost?*',
                                   '## 3. Quick look at the data (2-3 real numbers)\n'
                                   '\n'
                                   '*Load the starter CSV below and show 2-3 real numbers that make your lane look '
                                   'worth the next 7 weeks.*',
                                   "## 4. Careful words: what I can and can't claim\n"
                                   '\n'
                                   '*Write what your work will be able to say (observed, directional, '
                                   "decision-support) \u2014 and what it never will (causal proof, 'predicting "
                                   "Google').*"],
                           'selfcheck': '## Self-check\n'
                                        '\n'
                                        'Before you submit, confirm each line honestly:\n'
                                        '\n'
                                        '- [ ] Every section above is filled \u2014 markdown thinking AND the code that '
                                        'backs it\n'
                                        '- [ ] The notebook runs top to bottom with no errors (Runtime \u2192 Run all)\n'
                                        '- [ ] No client names, URLs, or private queries anywhere\n'
                                        '- [ ] My claims use careful words: observed, measured, directional, '
                                        'decision-support\n'
                                        '- [ ] Committed to my repo under `work/notebooks/` \u2014 then submit your repo '
                                        'URL on the card. Done.'},
 'w02_ml_task_framing': {'title': '# ML-03 \u2014 Frame Your Lane as an ML Task\n'
                                  '\n'
                                  '{BADGE}\n'
                                  '\n'
                                  'This skeleton is yours to fill. Work the sections **in order** \u2014 each one has a '
                                  'one-line hint. Simple words, honest numbers.\n'
                                  '\n'
                                  '> Working with an AI assistant? Tell it to read `skills/README.md` first and load '
                                  'the one skill this assignment names on its card.',
                         'sec': ['## 1. My lane as an ML task (type)\n'
                                 '\n'
                                 '*Classification, clustering, ranking, or scoring \u2014 which one, and why?*',
                                 '## 2. Target or proxy\n'
                                 '\n'
                                 '*What would you predict? Where does that label come from \u2014 observed outcome or a '
                                 'defined rule?*',
                                 "## 3. Success metric\n\n*One metric you can defend. What number means 'good'?*",
                                 '## 4. The unit of analysis, as a real dataframe\n'
                                 '\n'
                                 "*Load your lane's slice and show it: one row = one what?*",
                                 '## 5. Why ML beats a fixed rule here\n'
                                 '\n'
                                 '*What makes the pattern too messy for an if-statement?*'],
                         'selfcheck': '## Self-check\n'
                                      '\n'
                                      'Before you submit, confirm each line honestly:\n'
                                      '\n'
                                      '- [ ] Every section above is filled \u2014 markdown thinking AND the code that '
                                      'backs it\n'
                                      '- [ ] The notebook runs top to bottom with no errors (Runtime \u2192 Run all)\n'
                                      '- [ ] No client names, URLs, or private queries anywhere\n'
                                      '- [ ] My claims use careful words: observed, measured, directional, '
                                      'decision-support\n'
                                      '- [ ] Committed to my repo under `work/notebooks/` \u2014 then submit your repo '
                                      'URL on the card. Done.'},
 'w03_data_contract': {'title': '# ML-04 \u2014 Search Intelligence Data Contract\n'
                                '\n'
                                '{BADGE}\n'
                                '\n'
                                'This skeleton is yours to fill. Work the sections **in order** \u2014 each one has a '
                                'one-line hint. Simple words, honest numbers.\n'
                                '\n'
                                '> Working with an AI assistant? Tell it to read `skills/README.md` first and load '
                                'the one skill this assignment names on its card.',
                       'sec': ['## 1. Unit of analysis + time window\n'
                               '\n'
                               '*One row = one what, over which dates? State it, then verify it below.*',
                               '## 2. Fields: feature / label / context / excluded\n'
                               '\n'
                               '*Sort every field you plan to touch into these four buckets. Excluded needs a why.*',
                               '## 3. Verify it with queries (grain, counts, missing values, windows)\n'
                               '\n'
                               '*Every claim above gets a query cell here. A contract claim without a query next to '
                               'it is a guess.*',
                               '## 4. Data limits\n'
                               '\n'
                               '*What can this data never tell you? Unbalanced history, GSC-only early rows, window '
                               'overlaps.*'],
                       'selfcheck': '## Self-check\n'
                                    '\n'
                                    'Before you submit, confirm each line honestly:\n'
                                    '\n'
                                    '- [ ] Every section above is filled \u2014 markdown thinking AND the code that backs '
                                    'it\n'
                                    '- [ ] The notebook runs top to bottom with no errors (Runtime \u2192 Run all)\n'
                                    '- [ ] No client names, URLs, or private queries anywhere\n'
                                    '- [ ] My claims use careful words: observed, measured, directional, '
                                    'decision-support\n'
                                    '- [ ] Committed to my repo under `work/notebooks/` \u2014 then submit your repo URL '
                                    'on the card. Done.'},
 'w03_feature_leakage_check': {'title': '# ML-05 \u2014 Feature Vector and Leakage/Privacy Check\n'
                                        '\n'
                                        '{BADGE}\n'
                                        '\n'
                                        'This skeleton is yours to fill. Work the sections **in order** \u2014 each one '
                                        'has a one-line hint. Simple words, honest numbers.\n'
                                        '\n'
                                        '> Working with an AI assistant? Tell it to read `skills/README.md` first '
                                        'and load the one skill this assignment names on its card.',
                               'sec': ['## 1. Build the feature vector\n'
                                       '\n'
                                       '*Code that actually builds it \u2014 engineered features, categorical handling, '
                                       'fills.*',
                                       '## 2. Feature notes (meaning, missing, categorical, available-when?)\n'
                                       '\n'
                                       '*For each feature: what it means, how missing values are handled, and '
                                       'whether it exists BEFORE the moment you predict.*',
                                       '## 3. The leakage hunt\n'
                                       '\n'
                                       '*Attack your own features: label-derived columns, future windows, product '
                                       'flags. Show the test.*',
                                       '## 4. What I excluded and why\n'
                                       '\n'
                                       '*The list of fields you refused to use \u2014 with one line of why each.*'],
                               'selfcheck': '## Self-check\n'
                                            '\n'
                                            'Before you submit, confirm each line honestly:\n'
                                            '\n'
                                            '- [ ] Every section above is filled \u2014 markdown thinking AND the code '
                                            'that backs it\n'
                                            '- [ ] The notebook runs top to bottom with no errors (Runtime \u2192 Run '
                                            'all)\n'
                                            '- [ ] No client names, URLs, or private queries anywhere\n'
                                            '- [ ] My claims use careful words: observed, measured, directional, '
                                            'decision-support\n'
                                            '- [ ] Committed to my repo under `work/notebooks/` \u2014 then submit your '
                                            'repo URL on the card. Done.'},
 'w04_signal_audit': {'title': '# ML-06 \u2014 Signal Audit: Do the Flags Hold?\n'
                               '\n'
                               '{BADGE}\n'
                               '\n'
                               'This skeleton is yours to fill. Work the sections **in order** \u2014 each one has a '
                               'one-line hint. Simple words, honest numbers.\n'
                               '\n'
                               '> Working with an AI assistant? Tell it to read `skills/README.md` first and load '
                               'the one skill this assignment names on its card.',
                      'sec': ['## 1. Distributions\n'
                              '\n'
                              '*Look before deciding: distributions of your key fields. Note the heavy tails.*',
                              '## 2. Signal test #1 / #2 / #3 (verdict each)\n'
                              '\n'
                              '*Three safe signals, each with a mini-test and a verdict: CONFIRMED / OPPOSITE / '
                              'MIXED / FALSE.*',
                              '## 3. The flag-linked test\n'
                              '\n'
                              "*Pick a signal one of FlyRank's real flags relies on. Does the data support the "
                              "rule's assumption?*",
                              '## 4. What this means in practice\n'
                              '\n'
                              '*Two or three sentences: what a content team should take from this.*'],
                      'selfcheck': '## Self-check\n'
                                   '\n'
                                   'Before you submit, confirm each line honestly:\n'
                                   '\n'
                                   '- [ ] Every section above is filled \u2014 markdown thinking AND the code that backs '
                                   'it\n'
                                   '- [ ] The notebook runs top to bottom with no errors (Runtime \u2192 Run all)\n'
                                   '- [ ] No client names, URLs, or private queries anywhere\n'
                                   '- [ ] My claims use careful words: observed, measured, directional, '
                                   'decision-support\n'
                                   '- [ ] Committed to my repo under `work/notebooks/` \u2014 then submit your repo URL '
                                   'on the card. Done.'},
 'w04_baseline_score': {'title': '# ML-07 \u2014 Baseline Action Score and Top-20 Review\n'
                                 '\n'
                                 '{BADGE}\n'
                                 '\n'
                                 'This skeleton is yours to fill. Work the sections **in order** \u2014 each one has a '
                                 'one-line hint. Simple words, honest numbers.\n'
                                 '\n'
                                 '> Working with an AI assistant? Tell it to read `skills/README.md` first and load '
                                 'the one skill this assignment names on its card.',
                        'sec': ['## 1. My rule and its reason codes\n'
                                '\n'
                                '*Write the rule in plain words first. Then the reason codes it can output.*',
                                '## 2. Build the ranked queue (writes the CSV)\n'
                                '\n'
                                '*Code the score, rank everything, write work/outputs/baseline_action_score.csv.*',
                                '## 3. Top-20 review\n'
                                '\n'
                                '*For each of the top 20: action, reason code, confidence note, and what would make '
                                'it wrong.*',
                                '## 4. Weak picks + leakage check\n'
                                '\n'
                                '*Which picks look wrong and why? Confirm no product flags or future windows leaked '
                                'in.*'],
                        'selfcheck': '## Self-check\n'
                                     '\n'
                                     'Before you submit, confirm each line honestly:\n'
                                     '\n'
                                     '- [ ] Every section above is filled \u2014 markdown thinking AND the code that '
                                     'backs it\n'
                                     '- [ ] The notebook runs top to bottom with no errors (Runtime \u2192 Run all)\n'
                                     '- [ ] No client names, URLs, or private queries anywhere\n'
                                     '- [ ] My claims use careful words: observed, measured, directional, '
                                     'decision-support\n'
                                     '- [ ] Committed to my repo under `work/notebooks/` \u2014 then submit your repo URL '
                                     'on the card. Done.'},
 'w05_model': {'title': '# ML-08 \u2014 Capstone Modeling Lane\n'
                        '\n'
                        '{BADGE}\n'
                        '\n'
                        'This skeleton is yours to fill. Work the sections **in order** \u2014 each one has a one-line '
                        'hint. Simple words, honest numbers.\n'
                        '\n'
                        '> Working with an AI assistant? Tell it to read `skills/README.md` first and load the one '
                        'skill this assignment names on its card.',
               'sec': ['## 1. Method choice and why\n\n*Which method from the toolkit, and why it fits your lane.*',
                       '## 2. Split design\n'
                       '\n'
                       '*Grouped by client? Time-aware? Say why this split is honest for your question.*',
                       '## 3. Train + compare vs my baseline\n'
                       '\n'
                       '*Same data, same metric, same split as your Week-4 baseline. Show the table.*',
                       '## 4. Errors and interpretation\n'
                       '\n'
                       '*Where is the model wrong? What does it lean on? A short error analysis beats a big metric '
                       'table.*'],
               'selfcheck': '## Self-check\n'
                            '\n'
                            'Before you submit, confirm each line honestly:\n'
                            '\n'
                            '- [ ] Every section above is filled \u2014 markdown thinking AND the code that backs it\n'
                            '- [ ] The notebook runs top to bottom with no errors (Runtime \u2192 Run all)\n'
                            '- [ ] No client names, URLs, or private queries anywhere\n'
                            '- [ ] My claims use careful words: observed, measured, directional, decision-support\n'
                            '- [ ] Committed to my repo under `work/notebooks/` \u2014 then submit your repo URL on the '
                            'card. Done.'},
 'w06_validation_audit': {'title': '# ML-09 \u2014 Validation and Research Claim Audit\n'
                                   '\n'
                                   '{BADGE}\n'
                                   '\n'
                                   'This skeleton is yours to fill. Work the sections **in order** \u2014 each one has a '
                                   'one-line hint. Simple words, honest numbers.\n'
                                   '\n'
                                   '> Working with an AI assistant? Tell it to read `skills/README.md` first and '
                                   'load the one skill this assignment names on its card.',
                          'sec': ['## 1. Two paper findings + my methodology questions\n'
                                  '\n'
                                  '*Pick two findings from the FlyRank research paper. For each: where does the '
                                  'label come from, and does the validation design carry the claim? Constructive '
                                  'tone.*',
                                  '## 2. My model under an honest split (before/after)\n'
                                  '\n'
                                  '*Re-run your Week-5 model under a grouped or time-aware split. Show both '
                                  'numbers.*',
                                  '## 3. Leakage audit\n\n*The same hunt from Week 3, on your final feature set.*',
                                  '## 4. Claim rewrite\n'
                                  '\n'
                                  '*Take your own boldest sentence and rewrite it in safe language: observed, '
                                  'measured, directional, decision-support.*'],
                          'selfcheck': '## Self-check\n'
                                       '\n'
                                       'Before you submit, confirm each line honestly:\n'
                                       '\n'
                                       '- [ ] Every section above is filled \u2014 markdown thinking AND the code that '
                                       'backs it\n'
                                       '- [ ] The notebook runs top to bottom with no errors (Runtime \u2192 Run all)\n'
                                       '- [ ] No client names, URLs, or private queries anywhere\n'
                                       '- [ ] My claims use careful words: observed, measured, directional, '
                                       'decision-support\n'
                                       '- [ ] Committed to my repo under `work/notebooks/` \u2014 then submit your repo '
                                       'URL on the card. Done.'},
 'w07_action_playbook': {'title': '# ML-10 \u2014 Content Action Playbook\n'
                                  '\n'
                                  '{BADGE}\n'
                                  '\n'
                                  'This skeleton is yours to fill. Work the sections **in order** \u2014 each one has a '
                                  'one-line hint. Simple words, honest numbers.\n'
                                  '\n'
                                  '> Working with an AI assistant? Tell it to read `skills/README.md` first and load '
                                  'the one skill this assignment names on its card.',
                         'sec': ['## 1. Ranked actions + reason codes\n'
                                 '\n'
                                 '*The queue: what to do first, and why, in words a human trusts.*',
                                 '## 2. Intended use and limits\n'
                                 '\n'
                                 '*Who uses this, for what \u2014 and where it stops being valid.*',
                                 '## 3. Human review + the no-go list\n'
                                 '\n'
                                 '*What a person must check before acting. What should never be automated.*',
                                 '## 4. Monitoring / retrain triggers\n'
                                 '\n'
                                 '*What would tell you the recommendations went stale?*',
                                 '## 5. Exports for the paper\n'
                                 '\n'
                                 '*Write the queue (and any figures you want to reuse) to work/outputs/ \u2014 your paper '
                                 'builds on these files.*'],
                         'selfcheck': '## Self-check\n'
                                      '\n'
                                      'Before you submit, confirm each line honestly:\n'
                                      '\n'
                                      '- [ ] Every section above is filled \u2014 markdown thinking AND the code that '
                                      'backs it\n'
                                      '- [ ] The notebook runs top to bottom with no errors (Runtime \u2192 Run all)\n'
                                      '- [ ] No client names, URLs, or private queries anywhere\n'
                                      '- [ ] My claims use careful words: observed, measured, directional, '
                                      'decision-support\n'
                                      '- [ ] Committed to my repo under `work/notebooks/` \u2014 then submit your repo '
                                      'URL on the card. Done.'},
 'capstone': {'title': '# Capstone \u2014 mirrors your deployed research paper\n'
                       '\n'
                       '{BADGE}\n'
                       '\n'
                       'This skeleton is yours to fill. Work the sections **in order** \u2014 each one has a one-line '
                       'hint. Simple words, honest numbers.\n'
                       '\n'
                       '> Working with an AI assistant? Tell it to read `skills/README.md` first and load the one '
                       'skill this assignment names on its card.',
              'sec': ['## 1. Question\n\n*The research question and the decision it supports.*',
                      '## 2. Data\n'
                      '\n'
                      '*Which release, which tables, date windows, what you excluded and why. Public-safe.*',
                      '## 3. Methodology\n'
                      '\n'
                      '*Assumptions, features, label definition, baseline, validation design, leakage checks.*',
                      '## 4. Results (vs baseline)\n\n*Model vs baseline on the same split. The honest table.*',
                      '## 5. Limitations\n\n*What this work cannot claim.*',
                      '## 6. Ranked recommendations\n'
                      '\n'
                      "*The action playbook output \u2014 the paper's recommendations section.*",
                      '## 7. Artifacts the paper embeds\n'
                      '\n'
                      '*Generate/collect the charts and tables your deployed page will show.*'],
              'selfcheck': '## Self-check\n'
                           '\n'
                           'Before you submit, confirm each line honestly:\n'
                           '\n'
                           '- [ ] Every section above is filled \u2014 markdown thinking AND the code that backs it\n'
                           '- [ ] The notebook runs top to bottom with no errors (Runtime \u2192 Run all)\n'
                           '- [ ] No client names, URLs, or private queries anywhere\n'
                           '- [ ] My claims use careful words: observed, measured, directional, decision-support\n'
                           '- [ ] Committed to my repo under `work/notebooks/` \u2014 then submit your repo URL on the '
                           'card. Done.\n'
                           '- [ ] My deployed paper has **all 9 sections** \u2014 including the **Abstract** at the top '
                           'and **Acknowledgments & data credit** (the https://flyrank.ai link) at the bottom.\n'
                           "- [ ] **ML-12 done in this notebook's closing cells:** 5-minute demo outline + a "
                           'social-post cut + a 3-sentence employer-facing summary.\n'}}
