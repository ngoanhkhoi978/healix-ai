import torch, pprint
p = r"d:\workspace\python\Healix_AI\models\mri\model.pth"
state = torch.load(p, map_location='cpu')
print('TYPE', type(state))
if isinstance(state, dict):
    keys = list(state.keys())
    print('\nTop-level keys (first 50):')
    pprint.pprint(keys[:50])
    # show sample key types
    for k in keys[:20]:
        v = state[k]
        print('\nKEY:', k, '->', type(v))
        if isinstance(v, dict):
            print('  nested dict keys (first 20):', list(v.keys())[:20])
        elif hasattr(v, 'keys'):
            try:
                print('  has keys: ', list(v.keys())[:10])
            except Exception:
                pass
        else:
            try:
                print('  sample keys:', list(state.keys())[:10])
            except Exception:
                pass
else:
    # treat as state_dict
    print('\nAssuming state is a state_dict with sample keys:')
    pprint.pprint(list(state.keys())[:200])
print('\nDone')

