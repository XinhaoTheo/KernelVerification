"""Generate a browsable GLM index from canonical traces without any API calls."""
import json
from pathlib import Path
from summarize_traces import build_report, BENCHMARK
from case_registry import case_sort_key
from index_cases import DATASET_LABELS


def main():
    report=build_report()
    rows=[row for group in report['arms'].values() for row in group['per_case'].values()
          if row['path'].startswith('traces_glm/')]
    lines=['# GLM trace index','',
        'All GLM runs are listed together by case, arm and trial. API provenance is retained in trace metadata. '
        'Historical tool traces do not have raw API payloads; new runs include `llm_calls/`.', '',
        'See [案例总索引](../CASE_INDEX.md) for numeric ranges, original IDs and source kernels. '
        'Historical trace payloads keep their original IDs; the current directory and registry determine the case.', '']
    current_dataset = None
    for r in sorted(rows,key=lambda x:(case_sort_key(x['case']),x['arm'],x['trial'])):
        if r['dataset'] != current_dataset:
            current_dataset = r['dataset']
            lines += ['', '## ' + DATASET_LABELS.get(current_dataset, current_dataset), '',
                      '| Dataset | Case | Arm | Trial | Verdict / outcome | Trace |',
                      '| --- | --- | --- | --- | --- | --- |']
        path=Path(r['path']).relative_to('traces_glm')
        target=path/('transcript.md' if (BENCHMARK/r['path']/'transcript.md').exists() else 'trace_meta.json')
        lines.append(f"| {r['dataset']} | {r['case']} | {r['arm']} | {r['trial']} | "
                     f"{r.get('verdict')} / {r['outcome']} | [open]({target.as_posix()}) |")
    lines+=['',f'{len(rows)} recorded GLM trials. See [trace format](../TRACES.md) for provenance and capture limits.','']
    (BENCHMARK/'traces_glm/INDEX.md').write_text('\n'.join(lines))
    print(f'Indexed {len(rows)} GLM trials')

if __name__=='__main__':main()
