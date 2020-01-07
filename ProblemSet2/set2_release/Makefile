PY = time python
PLX = pdflatex
fname = ProblemSet2
help = ProblemSet2helper.py

all: python latex clean open

python: $(fname).py
	$(PY) $(fname).py

latex: $(fname).tex
	$(PLX) $(fname).tex

open:
	open $(fname).pdf

clean: 
	rm $(fname).aux $(fname).fdb_latexmk $(fname).fls $(fname).log $(fname).synctex.gz