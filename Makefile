PYENV := build/pyenv
DEPS := tensorflow==2.2.0rc4 tfjs-graph-converter==1.0.1

out.stamp: $(PYENV)/stamp
	./fetch-and-convert-all.bash iamamakefile
	touch out.stamp

$(PYENV)/stamp:
	mkdir -p $(PYENV)
	virtualenv -p $$(which python3) $(PYENV)
	$(PYENV)/bin/pip install tensorflow==2.2.0rc4 tfjs-graph-converter==1.0.1
	touch $(PYENV)/stamp

.PHONY: clean
clean:
	rm -rf build out
