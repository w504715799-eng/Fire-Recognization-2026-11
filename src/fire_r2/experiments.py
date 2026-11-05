"""Explicit factor matrix and dependency-free configuration checks."""
from copy import deepcopy
from .common import ProjectError

VARIANTS={
    'B_native':{'native':True},
    'B':{},'B+RuleW':{'weighting':'rule'},'B+W':{'weighting':'qwen'},
    'B+ShuffleW':{'weighting':'shuffle'},'B+WGlobal':{'weighting':'qwen','normalization':'global'},
    'B+P':{'semantic':'presence'},'B+S':{'semantic':'qwen'},
    'B+ShuffleS':{'semantic':'qwen','shuffle_semantic':True},
    'B+W+S':{'weighting':'qwen','semantic':'qwen'},
    'M-ordinary':{'m_mode':'ordinary'},'M-split':{'m_mode':'split'},'M-gate':{'m_mode':'gate'},
    'M':{'m_mode':'gain'},'M-noD':{'m_mode':'gain','detail':False},
    'M-noGain':{'m_mode':'gate'},'M-noSmokeAux':{'m_mode':'gain','smoke_aux':False},
    'M-fixedA':{'m_mode':'gain','fixed_split':True},'M+P':{'m_mode':'gain','semantic':'presence'},
    'M+W+P':{'m_mode':'gain','weighting':'qwen','semantic':'presence'},
    'B+SUP':{'sup_only':True},'M+W':{'m_mode':'gain','weighting':'qwen'},
    'M+S':{'m_mode':'gain','semantic':'qwen'},'M+W+S':{'m_mode':'gain','weighting':'qwen','semantic':'qwen'},
}


def factor_config(variant):
    if variant not in VARIANTS:raise ProjectError(f'Unknown experiment variant: {variant}')
    return {'weighting':'uniform','semantic':None,'m_mode':None,'width':64,'detail':True,
            'smoke_aux':True,'gain_weight':.05,'smoke_weight':.20,'semantic_warmup_epochs':5,
            'normalization':'group',**deepcopy(VARIANTS[variant])}


def experiment_matrix(base):
    result=[]
    for variant in VARIANTS:
        config=deepcopy(base);config['experiment']=variant
        safe=variant.replace('+','_').replace('-','_')
        config['run_name']=f"{safe}_{base['dataset']}_seed{base['seed']}"
        config['supervision']=f"artifacts/supervision/{safe}-{base['dataset']}.json"
        config['research']=factor_config(variant)
        result.append(config)
    return {'schema_version':1,'status':'configuration_menu_not_execution_request',
            'seed_repeats':[17,29,43],'training_sources':['dfire','homefire','fasdd_cv'],
            'evaluation_sources':['dfire','homefire','fasdd_cv','fasdd_uav'],'experiments':result}
