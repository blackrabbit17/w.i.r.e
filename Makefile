
export PYTHONPATH := $(shell pwd)/wire:$(PYTHONPATH)

train:
	python wire/main.py

m_train:
	python wire/main_multivariate.py
