
export PYTHONPATH := $(shell pwd)/wire:$(PYTHONPATH)

train:
	python wire/main.py
