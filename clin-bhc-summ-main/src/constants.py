import os

##############################################################
### directories ##############################################

# TODO: set DIR_PROJECT as location for all data and models
DIR_PROJECT = '../clin-bhc-summ/models/'
# 确保目录存在，虽然断言会报错，但最好手动检查路径是否正确
# assert os.path.exists(DIR_PROJECT), 'please enter valid directory'

# TODO: for best practice, move data to DIR_PROJECT/data/ (outside repo)
DIR_DATA = os.path.join(DIR_PROJECT, 'data/') # input data
if not os.path.exists(DIR_DATA):
    DIR_DATA = 'data/'

# directory of tuned models. created automatically
DIR_MODELS_TUNED = os.path.join(DIR_PROJECT, 'models_tuned/') # tuned models

# directory of physionet's pre-trained models (clin-t5, clin-t5-sci)
DIR_MODELS_CLIN = os.path.join(DIR_PROJECT, 'physionet.org/files/clinical-t5/1.0.0/')

##############################################################
### models (这是之前报错缺失的部分) ###########################

MODELS = {
    "llama2-13b": os.path.join(DIR_PROJECT, 'llama2'),  # Your local LLaMA2 path
    "llama3.1-8b": os.path.join(DIR_PROJECT, 'llama3.1'),
    "qwen2.5-14b": os.path.join(DIR_PROJECT, 'qwen2.5'), 
    "qwen3-8b": os.path.join(DIR_PROJECT, 'qwen3'),
    "t5-base": "t5-base",
    "flan-t5-base": "google/flan-t5-base",
    "scifive-base": "razent/SciFive-base-Pubmed_PMC", 
    "clin-t5-sci": os.path.join(DIR_MODELS_CLIN, 'Clinical-T5-Sci'),
    "clin-t5-base": os.path.join(DIR_MODELS_CLIN, 'Clinical-T5-Base'),
    "t5-large": "t5-large",
    "flan-t5-large": "google/flan-t5-large",
    "scifive-large": "razent/SciFive-large-Pubmed_PMC",
    "clin-t5-large": os.path.join(DIR_MODELS_CLIN, 'Clinical-T5-Large'),
    "flan-t5-xxl": "google/flan-t5-xxl",
    "flan-ul2": "google/flan-ul2",
    "falcon-7b-instruct": "tiiuae/falcon-7b-instruct",
}

##############################################################
### hyperparameters ##########################################

METHODS = ['lora', 'prefix_tuning']

MAX_LEN = 3100 
MAX_NEW_TOKENS = 260
PROMPT_LEN = 20 

# General Training Params
BATCH_SIZE_BASE = 4
BATCH_SIZE_PREFIX_TUNE = 4
BATCH_SIZE_LORA = 1  

LR0_PREFIX_TUNE = 1e-2
LR0_LORA = 2e-4  # learning rate

TRN_EPOCHS_PREFIX_TUNE = 15
TRN_EPOCHS_LORA = 10
PATIENCE = 10

# LoRA Params
LORA_R = 64 
LORA_ALPHA = 128 
LORA_DROPOUT = 0.1

##############################################################
### prompting ################################################

START_PREFIX = "You are an experienced physician. Based on the clinical findings, write a concise impression summarizing the key diagnoses and clinical significance."
ICL_PROMPT = '[ICL_PROMPT]'

DEFAULTS = {
    'method': 'discrete',
    'insert_prefix': '### Clinical Findings:\n',
    'prompt': '\n### Clinical Impression:\n',
    'grad_accum_steps': 16, 
    'lr_n_warmup_steps': 200,
    'lr_schedule': 'linear_decay', 
    'max_new_tokens': MAX_NEW_TOKENS,
}

cases = { 
    0: {},
    1: {'prompt': f'\n{ICL_PROMPT}_1\n### Clinical Impression:'},
    2: {'prompt': f'\n{ICL_PROMPT}_2\n### Clinical Impression:'},
    4: {'prompt': f'\n{ICL_PROMPT}_4\n### Clinical Impression:'},
    5: {'start_prefix': START_PREFIX},

    100: {'method': 'lora', 'batch_size': 4, 'grad_accum_steps': 16, 'trn_epochs': 20, 'lr0': 2e-4,'lora_dropout': 0.1},
    101: {'method': 'lora', 'batch_size': 4, 'grad_accum_steps': 16, 'trn_epochs': 20, 'lr0': 1e-4,'lora_dropout': 0.1},
    102: {'method': 'lora', 'batch_size': 2, 'grad_accum_steps': 32, 'trn_epochs': 20, 'lr0': 5e-5, 'lora_dropout': 0.05},
    103: {'method': 'lora', 'batch_size': 1, 'grad_accum_steps': 16, 'trn_epochs': 5, 'lr0': 5e-5, 'lora_dropout': 0.2},
    104: {'method': 'lora', 'batch_size': 1, 'grad_accum_steps': 16, 'trn_epochs': 5, 'lr0': 2e-4, 'lora_dropout': 0.2},
    105: {'method': 'lora', 'batch_size': 1, 'grad_accum_steps': 16, 'trn_epochs': 5, 'lr0': 2e-4, 'lora_dropout': 0.2},
    
    200: {'method': 'lora', 'batch_size': 1, 'grad_accum_steps': 16, 'trn_epochs': 15, 'lr0': 2e-4},
    201: {'method': 'lora', 'batch_size': 1, 'grad_accum_steps': 16, 'trn_epochs': 15, 'lr0': 2e-4},
    202: {'method': 'lora', 'batch_size': 1, 'grad_accum_steps': 16, 'trn_epochs': 15, 'lr0': 2e-4},
    203: {'method': 'lora', 'batch_size': 1, 'grad_accum_steps': 16, 'trn_epochs': 15, 'lr0': 2e-4},
    
    300: {'method': 'lora', 'batch_size': 1, 'grad_accum_steps': 16, 'trn_epochs': 15, 'lr0': 2e-4},
    301: {'method': 'lora', 'batch_size': 1, 'grad_accum_steps': 16, 'trn_epochs': 15, 'lr0': 2e-4},
    302: {'method': 'lora', 'batch_size': 1, 'grad_accum_steps': 16, 'trn_epochs': 15, 'lr0': 2e-4},
    303: {'method': 'lora', 'batch_size': 1, 'grad_accum_steps': 16, 'trn_epochs': 15, 'lr0': 2e-4},
    
    400: {'method': 'lora', 'batch_size': 1, 'grad_accum_steps': 16, 'trn_epochs': 15, 'lr0': 2e-4},
    401: {'method': 'lora', 'batch_size': 1, 'grad_accum_steps': 16, 'trn_epochs': 15, 'lr0': 2e-4},
    402: {'method': 'lora', 'batch_size': 1, 'grad_accum_steps': 16, 'trn_epochs': 15, 'lr0': 2e-4},
    403: {'method': 'lora', 'batch_size': 1, 'grad_accum_steps': 16, 'trn_epochs': 15, 'lr0': 2e-4},
    
    500: {'method': 'lora', 'batch_size': 1, 'grad_accum_steps': 16, 'trn_epochs': 15, 'lr0': 2e-4},
    501: {'method': 'lora', 'batch_size': 1, 'grad_accum_steps': 16, 'trn_epochs': 15, 'lr0': 2e-4},
    502: {'method': 'lora', 'batch_size': 1, 'grad_accum_steps': 16, 'trn_epochs': 15, 'lr0': 2e-4},
    503: {'method': 'lora', 'batch_size': 1, 'grad_accum_steps': 16, 'trn_epochs': 15, 'lr0': 2e-4}
}

def set_method_params(cases, case_id, param_str, val_prefix_tune, val_lora):
    ''' set parameters specific to methods if not explicitly specified in cases '''
    if param_str not in cases[case_id].keys():
        if cases[case_id]['method'] == 'prefix_tuning':
            cases[case_id][param_str] = val_prefix_tune 
        elif cases[case_id]['method'] == 'lora':
            cases[case_id][param_str] = val_lora
    return cases

# append DEFAULTS keys to cases[case_id]
for case_id in cases:
    for key in DEFAULTS:
        if key not in cases[case_id]:
            cases[case_id][key] = DEFAULTS[key]

    cases = set_method_params(cases, case_id, 'batch_size',
                              BATCH_SIZE_PREFIX_TUNE, BATCH_SIZE_LORA)
    cases = set_method_params(cases, case_id, 'trn_epochs',
                              TRN_EPOCHS_PREFIX_TUNE, TRN_EPOCHS_LORA)
    cases = set_method_params(cases, case_id, 'lr0',
                              LR0_PREFIX_TUNE, LR0_LORA)

##############################################################
### misc filenames ###########################################

FN_FINDING = 'finding.csv'
FN_SUM_REF = 'summary_ref.csv' 
FN_SUM_GEN = 'summary_gen.csv' 
FN_SUM_GEN_CLEAN = 'summary_gen_clean.csv' 
FN_METRICS = 'metrics.json' 
FN_METRICS_CLEAN = 'metrics_clean.json' 
DIR_ICL = os.path.join(DIR_PROJECT, 'icl')
EVAL_CXR_DATA = False