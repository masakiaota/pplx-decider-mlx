"""Build the evaluation sets (DATA/eval) and the calibration pool (DATA/calib) from the files of download.py.

    python bench/quality/build_sets.py [--data bench/quality/data]

Needs pyarrow (to read Parquet), for example `uv run --with pyarrow python bench/quality/build_sets.py`.
CPU only, about 30 s plus the VQAv2 image downloads (about 120 images, 15 MB).

Every line is {"id", "set", "request", "options": {"temperature", "execution_mode"}, "label", "meta"}: the request
as sent to the engine, the gold label of the source dataset and where the item came from. Image paths in requests
are relative to the JSONL file's folder (eval/images/, calib/images/). The selection is fixed by seeds, so the files
are identical on every machine; they are checked against expected.json (sha256) at the end.

Evaluation (eval/): boolq100 (the gate), boolq-val300, boolq-choice-score60, squad-choice60, squad-locate100,
squad-multi30-{isolated,separate}, vqa30, qasper-long30, features16-{separate,isolated}; additional sets
squad-locate200b, qasper-long40b, vqa60b. Calibration (calib/, for quantization only, never for evaluation):
calib-boolq-train2000 (1,945 after near-duplicate removal), calib-squad-train500, calib-squad-locate-train200,
calib-qasper-train20, calib-vqa20. check_sets.py verifies that the two do not overlap.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import urllib.request
from pathlib import Path

from common import DATA, HERE, USER_AGENT, VQA_REPO, VQA_REVISION, sha256_file, write_jsonl

GATE_SEED = 20261117   # boolq100 and the feature probes
SEED = 20261119        # everything else (each set uses SEED + k)
SCORE5 = ["Definitely no", "Probably no", "Unclear from the passage", "Probably yes", "Definitely yes"]
CHOICE3 = {"yes": "Yes", "no": "No", "unknown": "The passage does not say"}
TARGETS = [3000, 6000, 12000, 20000, 28000]  # Qasper text cut lengths in characters (~0.7K..7K tokens)
BOOLQ, SQUAD, QASPER = 'google/boolq', 'rajpurkar/squad_v2', 'allenai/qasper'


def rows(path):
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


def item(set_name, ident, request, label, meta, mode='separate'):
    return dict(id=ident, set=set_name, request=request, options=dict(temperature=1.0, execution_mode=mode),
                label=label, meta=meta)


def noul(q):
    return {"type": "noul", "instructions": q}


_n = lambda t: re.sub(r'\s+', ' ', t).strip().lower()


def fetch_image(row, path):
    """The row's image into path (kept when present). The dataset viewer's image URLs expire: re-run download.py
    --refresh-vqa if this fails."""
    if path.exists():
        return
    image = row['image'][0] if isinstance(row['image'], list) else row['image']
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(image['src'], headers={'User-Agent': USER_AGENT})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f'cannot download {path.name} ({e}); run download.py --refresh-vqa and build again')
    path.write_bytes(data)


class Builder:
    def __init__(self, data):
        self.data = Path(data)
        self.raw = self.data / 'raw'
        self.eval_dir, self.calib_dir = self.data / 'eval', self.data / 'calib'
        self.E, self.C = {}, {}

    def vqa_pages(self, *offsets):
        return [self.raw / f'vqa-rows-{o}.json' for o in offsets]

    # ------------------------------------------------------------------ boolq100 (gate) and feature probes
    def gate_and_features(self):
        val = rows(self.raw / 'boolq-validation.parquet')
        indices = random.Random(GATE_SEED).sample(range(len(val)), 100)
        boolq = [dict(index=i, label=val[i]['answer'],
                      request={"state": val[i]['passage'], "questions": {"answer": noul(val[i]['question'])}}) for i in indices]
        yes = sum(b['label'] for b in boolq)
        assert 15 <= yes <= 85, f'extreme label imbalance: {yes}/100 yes'
        self.E['boolq100'] = [item('boolq100', f'boolq-validation-{b["index"]}', b['request'], b['label'],
                                   dict(dataset=BOOLQ, split='validation', index=b['index'])) for b in boolq]
        self.gate = set(indices)

        features = []   # (id, request, label, meta)
        source = boolq[0]
        bmeta = dict(dataset=BOOLQ, split='validation', index=source['index'])
        question = source['request']['questions']['answer']['instructions']
        decisions = {
            "noul": {"type": "noul", "instructions": question},
            "choice": {"type": "choice", "instructions": question, "criteria": {"yes": "Yes", "no": "No"}},
            "score": {"type": "score", "instructions": question, "criteria": ["No", "Yes"]},
        }
        passage = source['request']['state']
        features.append(('boolq-mixed-decisions', {"state": passage, "questions": decisions}, source['label'], bmeta))
        features.append(('boolq-object-state', {"state": {"passage": passage}, "questions": decisions}, source['label'], bmeta))
        features.append(('boolq-array-state', {"state": [passage], "questions": decisions}, source['label'], bmeta))
        features.append(('boolq-multiple-questions', {"state": passage, "questions": {f"answer_{i}": decisions["noul"] for i in range(4)}},
                         source['label'], bmeta))
        longest = max(boolq, key=lambda b: len(b['request']['state']))
        features.append(('boolq-longest-fixed-sample', longest['request'], longest['label'],
                         dict(dataset=BOOLQ, split='validation', index=longest['index'])))

        squad = rows(self.raw / 'squad_v2-validation.parquet')
        for number, index in enumerate(random.Random(GATE_SEED).sample(range(len(squad)), 6)):
            row = squad[index]
            questions = {"answer": {"type": "locate", "instructions": row["question"]}}
            meta = dict(dataset=SQUAD, split='validation', id=row['id'], index=index)
            features.append((f"squad-locate-{row['id']}", {"state": row["context"], "questions": questions}, None, meta))
            if number == 0:
                mixed = {**questions, "has_answer": {"type": "noul", "instructions": f"Does this passage answer the question: {row['question']}"}}
                features.append((f"squad-mixed-{row['id']}", {"state": row["context"], "questions": mixed}, None, meta))

        vqa, seen = [], set()
        for entry in json.loads(self.vqa_pages(0)[0].read_text())['rows']:
            row = entry['row']
            if row['image_id'] in seen:
                continue
            seen.add(row['image_id'])
            path = self.eval_dir / 'images' / f"vqa-{row['image_id']}.jpg"
            fetch_image(row, path)
            vqa.append(dict(row=row, row_idx=entry['row_idx'], image=f'images/{path.name}', sha=sha256_file(path)))
            if len(vqa) == 3:
                break
        first = None
        for v in vqa:
            row = v['row']
            correct = row['multiple_choice_answer']
            options = list(dict.fromkeys([correct] + [o['row']['multiple_choice_answer'] for o in vqa]))
            if len(options) < 2:
                options += ["none"]
            questions = {
                "choice": {"type": "choice", "instructions": row["question"], "criteria": {o: o for o in options}},
                "noul": {"type": "noul", "instructions": f"For the question '{row['question']}', is the answer '{correct}'?"},
                "score": {"type": "score", "instructions": f"For the question '{row['question']}', is the answer '{correct}'?", "criteria": ["No", "Yes"]},
            }
            meta = dict(dataset=VQA_REPO, split='validation', row_idx=v['row_idx'], image_id=row['image_id'],
                        question_id=row['question_id'], image_sha256=v['sha'])
            features.append((f"vqa-{row['question_id']}", {"state": "", "images": [v['image']], "questions": questions}, None, meta))
            first = first or features[-1]
        if len(vqa) > 1:
            ident, request, label, meta = first
            features.append((ident + '-multiple-images',
                             {**request, "images": [v['image'] for v in vqa[:2]], "state": "Answer the questions about the first image."},
                             label, meta))
        for mode in ('separate', 'isolated'):
            name = f'features16-{mode}'
            self.E[name] = [item(name, ident + ('' if mode == 'separate' else '@isolated'), request, label, meta, mode=mode)
                            for ident, request, label, meta in features]

    # ------------------------------------------------------------------ BoolQ
    def boolq(self):
        val = rows(self.raw / 'boolq-validation.parquet')
        train = rows(self.raw / 'boolq-train.parquet')
        rng = random.Random(SEED)
        pool = [i for i in range(len(val)) if i not in self.gate]
        rng.shuffle(pool)
        bq_noul, bq_cs = pool[:300], pool[300:360]
        meta = lambda i: dict(dataset=BOOLQ, split='validation', index=i)
        self.E['boolq-val300'] = [item('boolq-val300', f'boolq-validation-{i}',
                                       {"state": val[i]['passage'], "questions": {"answer": noul(val[i]['question'])}},
                                       bool(val[i]['answer']), meta(i)) for i in sorted(bq_noul)]
        self.E['boolq-choice-score60'] = [item('boolq-choice-score60', f'boolq-validation-{i}-cs',
                                               {"state": val[i]['passage'], "questions": {
                                                   "choice": {"type": "choice", "instructions": val[i]['question'], "criteria": CHOICE3},
                                                   "score": {"type": "score", "instructions": val[i]['question'], "criteria": SCORE5}}},
                                               bool(val[i]['answer']), meta(i)) for i in sorted(bq_cs)]
        # Calibration: BoolQ train without the rows whose passage is also in validation (15% of train) and without
        # near duplicates (the same paragraph in a longer or shorter version, or with the same 120-character start).
        rng = random.Random(SEED + 1)
        val_passages = {_n(r['passage']) for r in val}
        val_prefix = {p[:120] for p in val_passages}
        val_long = [p for p in val_passages if len(p) > 200]

        def near_val(t):
            t = _n(t)
            return t in val_passages or t[:120] in val_prefix or any(v in t or (len(t) > 200 and t in v) for v in val_long)

        tidx = [i for i in range(len(train)) if _n(train[i]['passage']) not in val_passages]
        rng.shuffle(tidx)
        self.C['calib-boolq-train2000'] = [item('calib-boolq-train2000', f'boolq-train-{i}',
                                                {"state": train[i]['passage'], "questions": {"answer": noul(train[i]['question'])}},
                                                bool(train[i]['answer']), dict(dataset=BOOLQ, split='train', index=i))
                                           for i in sorted(tidx[:2000]) if not near_val(train[i]['passage'])]

    # ------------------------------------------------------------------ SQuAD v2
    def squad(self):
        sv = rows(self.raw / 'squad_v2-validation.parquet')
        st = rows(self.raw / 'squad_v2-train.parquet')

        def spans(r):
            a = r['answers']
            return [{"start": int(s), "end": int(s) + len(t), "text": t} for t, s in zip(a['text'], a['answer_start'])]

        vmeta = lambda i: dict(dataset=SQUAD, split='validation', id=sv[i]['id'], index=i)
        rng = random.Random(SEED + 2)
        ans = [i for i, r in enumerate(sv) if r['answers']['text']]
        una = [i for i, r in enumerate(sv) if not r['answers']['text']]
        rng.shuffle(ans)
        rng.shuffle(una)
        loc = sorted(ans[:70] + una[:30])
        self.E['squad-locate100'] = [item('squad-locate100', f'squad-{sv[i]["id"]}-locate',
                                          {"state": sv[i]['context'], "questions": {"answer": {"type": "locate", "instructions": sv[i]['question']}}},
                                          spans(sv[i]), vmeta(i)) for i in loc]
        used = set(loc)
        # choice: the gold answer and 3 answers of other questions on the same article
        by_title = {}
        for i in ans:
            by_title.setdefault(sv[i]['title'], []).append(i)
        choice_items = []
        for i in ans[70:]:
            if len(choice_items) >= 60:
                break
            gold = sv[i]['answers']['text'][0]
            others = []
            for j in by_title[sv[i]['title']]:
                t = sv[j]['answers']['text'][0]
                if j != i and t.lower() not in gold.lower() and gold.lower() not in t.lower() and t not in others:
                    others.append(t)
            if len(others) < 3:
                continue
            r2 = random.Random(f'{SEED}-{sv[i]["id"]}')
            opts = [gold] + r2.sample(others, 3)
            r2.shuffle(opts)
            keys = ['a', 'b', 'c', 'd']
            choice_items.append(item('squad-choice60', f'squad-{sv[i]["id"]}-choice',
                                     {"state": sv[i]['context'], "questions": {"answer": {"type": "choice", "instructions": sv[i]['question'], "criteria": dict(zip(keys, opts))}}},
                                     keys[opts.index(gold)], vmeta(i)))
            used.add(i)
        self.E['squad-choice60'] = sorted(choice_items, key=lambda x: x['meta']['index'])
        # multi-question: one context, 2 answerable + 2 unanswerable "does this passage answer" questions
        by_ctx = {}
        for i, r in enumerate(sv):
            by_ctx.setdefault(r['context'], []).append(i)
        ctxs = [c for c, ix in by_ctx.items() if len(ix) >= 4 and not used.intersection(ix)
                and any(sv[k]['answers']['text'] for k in ix) and any(not sv[k]['answers']['text'] for k in ix)]
        rng = random.Random(SEED + 3)
        rng.shuffle(ctxs)
        multi = []
        for c in ctxs[:30]:
            ix = by_ctx[c]
            a = [k for k in ix if sv[k]['answers']['text']]
            u = [k for k in ix if not sv[k]['answers']['text']]
            pick = sorted((a[:2] + u[:2]) if len(a) >= 2 and len(u) >= 2 else ix[:4])
            qs = {f"q{n}": noul(f"Does this passage answer the question: {sv[k]['question']}") for n, k in enumerate(pick)}
            lab = {f"q{n}": bool(sv[k]['answers']['text']) for n, k in enumerate(pick)}
            meta = dict(dataset=SQUAD, split='validation', ids=[sv[k]['id'] for k in pick], index=pick)
            multi.append((pick[0], qs, lab, meta, c))
        multi.sort()
        for mode in ('isolated', 'separate'):
            self.E[f'squad-multi30-{mode}'] = [item(f'squad-multi30-{mode}', f'squad-multi-{m[3]["ids"][0]}-{mode}',
                                                    {"state": m[4], "questions": m[1]}, m[2], m[3], mode=mode) for m in multi]
        # calibration: SQuAD v2 train
        rng = random.Random(SEED + 4)
        tix = list(range(len(st)))
        rng.shuffle(tix)
        tmeta = lambda i: dict(dataset=SQUAD, split='train', id=st[i]['id'], index=i)
        self.C['calib-squad-train500'] = [item('calib-squad-train500', f'squad-train-{st[i]["id"]}',
                                               {"state": st[i]['context'], "questions": {"has_answer": noul(f"Does this passage answer the question: {st[i]['question']}")}},
                                               bool(st[i]['answers']['text']), tmeta(i)) for i in sorted(tix[:500])]
        self.C['calib-squad-locate-train200'] = [item('calib-squad-locate-train200', f'squad-train-{st[i]["id"]}-locate',
                                                      {"state": st[i]['context'], "questions": {"answer": {"type": "locate", "instructions": st[i]['question']}}},
                                                      spans(st[i]), tmeta(i)) for i in sorted(tix[500:700])]
        # additional Locate set: 140 answerable + 60 unanswerable, contexts not used by any core evaluation set
        used_ctx = {r['request']['state'] for rs in self.E.values() for r in rs if isinstance(r['request'].get('state'), str)}
        rng = random.Random(SEED + 20)
        idx = [i for i, r in enumerate(sv) if r['context'] not in used_ctx]
        rng.shuffle(idx)
        pick = sorted([i for i in idx if sv[i]['answers']['text']][:140] + [i for i in idx if not sv[i]['answers']['text']][:60])
        self.E['squad-locate200b'] = [item('squad-locate200b', f'squad-{sv[i]["id"]}-locate',
                                           {"state": sv[i]['context'], "questions": {"answer": {"type": "locate", "instructions": sv[i]['question']}}},
                                           spans(sv[i]), vmeta(i)) for i in pick]

    # ------------------------------------------------------------------ Qasper (long inputs)
    def qasper_items(self, papers, split, set_name, n, seed, targets, exclude_questions=()):
        cands = []
        for p in papers:
            q = p['qas']
            for question, qid, answers in zip(q['question'], q['question_id'], q['answers']):
                yn = [a['yes_no'] for a in answers['answer'] if a['yes_no'] is not None and not a['unanswerable']]
                if yn and all(v == yn[0] for v in yn) and len(yn) == len(answers['answer']):
                    cands.append((p, question, qid, yn[0]))
        rng = random.Random(seed)
        rng.shuffle(cands)
        out, seen = [], set()
        for p, question, qid, label in cands:
            if p['id'] in seen or _n(question) in exclude_questions:
                continue
            seen.add(p['id'])
            text = p['title'] + '\n\n' + p['abstract'] + '\n\n' + '\n\n'.join(
                f"{s}\n" + '\n'.join(par) for s, par in zip(p['full_text']['section_name'], p['full_text']['paragraphs']))
            target = targets[len(out) % len(targets)]
            if len(text) < target * 0.6:
                continue
            cut = text[:target]
            cut = cut[:cut.rfind('\n')] if '\n' in cut[target // 2:] else cut
            out.append(item(set_name, f'qasper-{qid}', {"state": cut, "questions": {"answer": noul(question)}}, bool(label),
                            dict(dataset=QASPER, split=split, paper=p['id'], question_id=qid, chars=len(cut))))
            if len(out) >= n:
                break
        return out

    def qasper(self):
        val = rows(self.raw / 'qasper-validation.parquet')
        self.E['qasper-long30'] = self.qasper_items(val, 'validation', 'qasper-long30', 30, SEED + 5, TARGETS)
        vq = {_n(q) for p in val for q in p['qas']['question']}
        self.C['calib-qasper-train20'] = self.qasper_items(rows(self.raw / 'qasper-train.parquet'), 'train',
                                                           'calib-qasper-train20', 20, SEED + 6, TARGETS, vq)
        # additional long set: papers not in qasper-long30, cut to 8K..28K characters (asks for 60, 40 qualify)
        used = {r['meta']['paper'] for r in self.E['qasper-long30']}
        self.E['qasper-long40b'] = self.qasper_items([p for p in val if p['id'] not in used], 'validation', 'qasper-long40b',
                                                     60, SEED + 30, [8000, 14000, 20000, 28000])

    # ------------------------------------------------------------------ VQAv2 (images)
    def vqa_items(self, data, set_name, n, seed, folder, prefix):
        rs = [r['row'] for r in data]
        by_type = {}
        for r in rs:
            by_type.setdefault(r['answer_type'], set()).add(r['multiple_choice_answer'])
        out, seen = [], set()
        rng = random.Random(seed)
        for k, r in enumerate(data):
            row = r['row']
            if row['image_id'] in seen:
                continue
            gold = row['multiple_choice_answer']
            others = sorted(by_type[row['answer_type']] - {gold})
            if len(others) < 3:
                others = sorted(set().union(*by_type.values()) - {gold})
            seen.add(row['image_id'])
            img = folder / 'images' / f'{prefix}-{row["image_id"]}.jpg'
            fetch_image(row, img)
            opts = [gold] + rng.sample(others, 3)
            rng.shuffle(opts)
            keys = ['a', 'b', 'c', 'd']
            # half of the binary probes ask about the gold answer, half about a wrong option (the first line only
            # advances the random state as the sets were built with it)
            probe = gold if k % 2 == 0 else rng.choice(opts[:1] + [o for o in opts if o != gold])
            probe = gold if k % 2 == 0 else next(o for o in opts if o != gold)
            out.append(item(set_name, f'vqa-{row["question_id"]}', {
                "state": "Answer the question about the image.", "images": [f'images/{img.name}'],
                "questions": {"choice": {"type": "choice", "instructions": row['question'], "criteria": dict(zip(keys, opts))},
                              "noul": noul(f"For the question '{row['question']}', is the answer '{probe}'?")}},
                {"choice": keys[opts.index(gold)], "noul": probe == gold},
                dict(dataset=VQA_REPO, split='validation', row_idx=r['row_idx'], image_id=row['image_id'],
                     question_id=row['question_id'], image_sha256=hashlib.sha256(img.read_bytes()).hexdigest())))
            if len(out) >= n:
                break
        return out

    def vqa(self):
        pages = lambda *o: [r for p in self.vqa_pages(*o) for r in json.loads(p.read_text())['rows']]
        self.E['vqa30'] = self.vqa_items(pages(1000, 1100), 'vqa30', 30, SEED + 7, self.eval_dir, 'eval')
        self.C['calib-vqa20'] = self.vqa_items(pages(100000, 100060), 'calib-vqa20', 20, SEED + 8, self.calib_dir, 'calib')
        used = {r['meta']['image_id'] for r in self.E['vqa30'] + self.C['calib-vqa20']}
        self.E['vqa60b'] = self.vqa_items([r for r in pages(1200, 1300, 1400, 1500) if r['row']['image_id'] not in used],
                                          'vqa60b', 60, SEED + 31, self.eval_dir, 'eval')

    def build(self):
        self.gate_and_features()
        self.boolq()
        self.squad()     # after boolq/gate: squad-locate200b avoids contexts of the core sets built so far
        self.qasper()
        self.vqa()
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        self.calib_dir.mkdir(parents=True, exist_ok=True)
        written = {}
        for folder, sets in ((self.eval_dir, self.E), (self.calib_dir, self.C)):
            for name, items in sets.items():
                path = folder / f'{name}.jsonl'
                write_jsonl(path, items)
                written[f'{folder.name}/{path.name}'] = sha256_file(path)
                print(f'{folder.name}/{path.name:36s} {len(items):5d}')
        return written


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', type=Path, default=DATA)
    ap.add_argument('--update-expected', action='store_true', help='maintainers: record the sha256 of the built files')
    a = ap.parse_args()
    b = Builder(a.data)
    written = b.build()
    manifest = dict(seeds=dict(gate=GATE_SEED, sets=SEED), vqa_revision=VQA_REVISION,
                    eval={k: len(v) for k, v in b.E.items()}, calib={k: len(v) for k, v in b.C.items()}, sha256=written)
    (a.data / 'build-manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    expected_path = HERE / 'expected.json'
    if a.update_expected:
        expected_path.write_text(json.dumps(dict(sha256=written), indent=1) + '\n')
        print(f'wrote {expected_path}')
        return
    expected = json.loads(expected_path.read_text())['sha256']
    bad = sorted(k for k in expected if written.get(k) != expected[k])
    if bad:
        raise SystemExit(f'{len(bad)} files differ from expected.json (changed sources or images?): {", ".join(bad)}')
    print(f'all {len(expected)} files match expected.json')


if __name__ == '__main__':
    main()
