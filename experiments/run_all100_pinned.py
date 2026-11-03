import sys, json, glob, os
REPO = '/Users/amdjed/Downloads/cardiac-mri-grounded-vlm'
sys.path[:0] = [REPO + '/report_generation', REPO]
import config
import run_region_grounded_poc as r
cases = sorted(os.path.splitext(os.path.basename(f))[0] for f in glob.glob('patients/*.json'))
assert len(cases) == 100, len(cases)
print('settings:', config.GEMINI_BACKEND, config.GEMINI_MODEL, config.GEMINI_TEMPERATURE, config.GEMINI_SEED, flush=True)
r.run_all(cases)
bad = []
for c in cases:
    rec = json.load(open(f'outputs_region_grounded/{c}.json'))
    gs = rec.get('generation_settings') or {}
    if rec.get('model') != config.GEMINI_MODEL or gs.get('backend') != 'vertex':
        bad.append((c, rec.get('model'), gs))
print('records:', len(glob.glob('outputs_region_grounded/*.json')), 'off-config:', bad, flush=True)
print('ALL_DONE', flush=True)
