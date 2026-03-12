.PHONY: compile test toy native-help help

help:
	@echo "make compile  - compile Python modules"
	@echo "make test     - run test suite"
	@echo "make toy      - run the toy pipeline"
	@echo "make native-help - show native engine CLI"

compile:
	python3 -m compileall nano_verl tests

test:
	python3 -m unittest discover -s tests

toy:
	python3 -m nano_verl.main --num-samples 4 --strategy best_of_n

native-help:
	python3 -m nano_verl.train_native --help
