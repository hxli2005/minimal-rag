"""Freeze disjoint samples and byte-identical pilot reuse before any model call."""
import random
import shutil
import subprocess
from shared import HERE, ROOT, OUT, PILOT, pilot, queries, read, save, note


def main():
    all_ids = sorted(queries())
    assert len(all_ids) == 830
    old = pilot.sample()
    extra = random.Random(20261016).sample(sorted(set(all_ids) - set(old)), 70)
    dev = old + extra
    remaining = sorted(set(all_ids) - set(dev))
    b200 = random.Random(20261017).sample(remaining, 200)
    for name, values, seed in [('pilot30', old, 20261015), ('dev100', dev, 20261016),
                                ('new70', extra, 20261016), ('b200', b200, 20261017),
                                ('remaining730', remaining, None), ('all830', all_ids, None)]:
        value = {'query_ids': values, 'seed': seed, 'sampling_population_sorted_as_strings': True}
        path = HERE / f'sample-{name}.json'
        assert read(path, value) == value
        save(path, value)
    assert not set(dev) & set(b200)
    source_config = pilot.OUT / 'config.json'
    target = HERE / 'pilot-reference-config.json'
    if target.exists():
        assert target.read_bytes() == source_config.read_bytes()
    else:
        shutil.copyfile(source_config, target)
    # Existing pilot commit provides the baseline; no source copies or new hashes.
    unchanged = subprocess.run(['git', 'diff', '--exit-code', '1dad042', '--', 'src'],
                               cwd=ROOT, capture_output=True).returncode == 0
    assert unchanged, 'Core sources changed since the pilot'
    reused = 0
    for qid in old:
        for kind in ['traces', 'official', 'judgements']:
            source = pilot.OUT / kind / f'{qid}.json'
            dest = OUT / 'A' / kind / source.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                shutil.copyfile(source, dest)
            assert source.read_bytes() == dest.read_bytes()
            reused += 1
    save(HERE / 'reuse-validation.json', {'pilot_configuration_byte_identical': True,
         'source_unchanged_since_pilot_commit': '1dad042', 'copied_files_byte_identical': reused,
         'reused_query_ids': old, 'reused_generation_requests': pilot.request_count(),
         'pilot_requests_excluded_from_new_stage_budgets': True})
    save(HERE / 'selection-rule.json', {'only_candidate': 'A-lean', 'excerpt_chars': 800,
         'minimum_uncached_token_reduction': .40, 'maximum_losses_minus_wins': 2,
         'require_complete_paired_dev100_and_usage': True, 'fallback': 'A'})
    note('决定：按字符串排序抽样并冻结dev100/b200，A复用试点30题的90份逐字节一致文件；理由：避免重跑和样本泄漏，新增阶段预算不重复计入历史289次调用。')
    print({'dev100': len(dev), 'b200': len(b200), 'remaining': len(remaining), 'reused_files': reused})


if __name__ == '__main__':
    main()
