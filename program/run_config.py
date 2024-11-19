NUMBER_OF_RUNS = 10

# Available datasets : {'askgit', 'sof', 'ubuntu', 'banking', 'msa', 'msr'}
DATASET = 'askgit'

TRAINING_DATA_PERCENT = 0.30
TESTING_DATA_PERCENT = 0.50
SEED = 1234
USE_SEED = True

# percent of expanded data to be used for the pruner
PRUNER_DATA_PERCENT = 0.4

SEM_GROUP_THRESHOLD = 0.8
TRANSFORMER = 'sentence-transformers/sentence-t5-xxl'

MAX_WORDS_PER_LF = 3

OUTPUT_PATH = './results/' + DATASET + '/'
LOGS_PATH = './logs/' + DATASET + '/'

WORDS_LABELLER_THRESHOLD = 0.8
ENTITY_LABELLER_THRESHOLD = 0.8
WORDS_ENTITY_LABELLER_THRESHOLD = 0.8

PRUNER_SCORE_THRESHOLD = 0.3
MODEL_LOCATION = ''