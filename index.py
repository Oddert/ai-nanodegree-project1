import numpy as np

from datasets import load_dataset
from loguru import logger
from peft import (
	AutoPeftModelForCausalLM,
	get_peft_model,
	LoraConfig,
	TaskType,
)
from transformers import (
	AutoTokenizer,
	# DistilBertForSequenceClassification,
	BertForSequenceClassification,
	Trainer,
	TrainingArguments,
)

# TODO: Would have preferred to use a more creative dataset but ran into too many problems to keep up with the course. 
model_name = 'bert-base-cased'
# model_name = 'distilbert-base-uncased'

# dataset_name = 'dair-ai/emotion'
# dataset_name = 'roman_urdu_hate_speech'
# dataset_name = 'ctoraman/gender-hate-speech'
dataset_name = 'imdb'

tokeniser_key = 'text'
train_key = 'train'
test_key = 'test'
select_size = 250
mode_save_name = 'bert-lora'
dir_save_name = './data/project1/initial'

dataset = load_dataset(dataset_name)

tokeniser = AutoTokenizer.from_pretrained(model_name)

def tokenise(examples):
	return tokeniser(
        examples[tokeniser_key],
        padding='max_length',
        truncation=True
    )

tokenised_ds = dataset.map(tokenise, batched=True)

sampled_ds = {}

for label in [train_key, test_key]:
      sampled_ds[label] = tokenised_ds[label].shuffle(seed=202405241534).select(range(select_size))

# train_ds = train_ds.rename_column('Text', 'text')
# train_ds = train_ds.rename_column('Label', 'label')

# test_ds = test_ds.rename_column('Text', 'text')
# test_ds = test_ds.rename_column('Label', 'label')

lora_config = LoraConfig(
	task_type=TaskType.SEQ_CLS,
	r=1,
	lora_alpha=1,
	lora_dropout=0.1,
)

### Loading and Evaluating a Foundation Model
## Loading the model
# Once you have selected a model, load it in your notebook.
base_model = BertForSequenceClassification.from_pretrained(
	model_name,
	num_labels=2,
)
print(base_model)
logger.info('base model instantiated')
# base_model = DistilBertForSequenceClassification.from_pretrained(
# 	model_name,
# 	num_labels=2,
# )

# TODO: Copied from earlier lesson, potentially upgrade
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy': (predictions == labels).mean()}

training_args = TrainingArguments(
	output_dir='./data/project1/initial',
	evaluation_strategy='epoch',
	num_train_epochs=1,
)

trainer = Trainer(
	args=training_args,
	compute_metrics=compute_metrics,
	model=base_model,
	eval_dataset=sampled_ds[test_key],
	train_dataset=sampled_ds[train_key],
)
logger.info('initial trainer instantiated')

## Evaluating the model
# Perform an initial evaluation of the model on your chosen sequence classification task. This step will require that you also load an appropriate tokenizer and dataset.
initial_evaluation = trainer.evaluate()
logger.info('initial trainer evaluation complete')
logger.info('The base model was evaluated with the following metrics:')
logger.info(initial_evaluation)

### Performing Parameter-Efficient Fine-Tuning
## Creating a PEFT config
# Create a PEFT config with appropriate hyperparameters for your chosen model.
peft_config = LoraConfig()
logger.info('peft config created')

## Creating a PEFT model
# Using the PEFT config and foundation model, create a PEFT model.
model = get_peft_model(base_model, peft_config)
model.print_trainable_parameters()
logger.info('lora model instantiated')

peft_training_args = TrainingArguments(
	output_dir=dir_save_name,
	evaluation_strategy='epoch',
	num_train_epochs=10,
)

peft_trainer = Trainer(
	args=peft_training_args,
	compute_metrics=compute_metrics,
	model=model,
	eval_dataset=sampled_ds[test_key],
	train_dataset=sampled_ds[train_key],
)
logger.info('peft trainer instantiated')

## Training the model
# Using the PEFT model and dataset, run a training loop with at least one epoch.
peft_trainer.train()
logger.info('peft trainer training complete')

## Saving the trained model
# Depending on your training loop configuration, your PEFT model may have already been saved. If not, use save_pretrained to save your progress.
logger.info('beginning save...')
model.save_pretrained(mode_save_name)
logger.info('save complete.')

### Performing Inference with a PEFT Model
## Loading the model
# Using the appropriate PEFT model class, load your trained model.
logger.info('loading saved model...')
final_model = AutoPeftModelForCausalLM.from_pretrained(mode_save_name)
logger.info('...model loaded')
logger.info(final_model)

## Evaluating the model
# Repeat the previous evaluation process, this time using the PEFT model. Compare the results to the results from the original foundation model.
final_trainer = Trainer(
	args=peft_training_args,
	compute_metrics=compute_metrics,
	model=final_model,
	eval_dataset=sampled_ds[test_key],
	train_dataset=sampled_ds[train_key],
)
logger.info('final trainer instantiated')

final_evaluation = final_trainer.evaluate()
logger.info('evaluation complete :)')

logger.info('The final model was evaluated with the following metrics:')
logger.info(final_evaluation)
