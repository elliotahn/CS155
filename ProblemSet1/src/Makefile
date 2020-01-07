PY = time python
PLX = pdflatex
fname = ProblemSet1
sgdp = sgdProblem.py

all: python sgd latex clean open

sgd: $(sgdp)
	$(PY) $(sgdp)

python: $(fname).py
	$(PY) $(fname).py

latex: $(fname).tex
	$(PLX) $(fname).tex

open:
	open $(fname).pdf

clean: 
	rm $(fname).aux $(fname).fdb_latexmk $(fname).fls $(fname).log $(fname).synctex.gz