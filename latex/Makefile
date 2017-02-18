#!/usr/local/bin/zsh

COMPFILES = *.pdf *.aux *.log *.idx *.toc *.bbl *.blg

all: main.pdf

main.pdf: main.tex $(wildcard chapters/*.tex)
	pdflatex $(basename $<) 
	bibtex $(basename $<) 
	makeindex $(basename $<)
	pdflatex $(basename $<) 
	pdflatex $(basename $<) 
	mv $(basename $<).pdf $@
	open main.pdf

clean: 
	rm -rf $(COMPFILES)
	rm inputs.tex
