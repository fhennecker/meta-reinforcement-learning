COMPFILES = *.pdf *.aux *.log *.idx *.toc *.bbl *.blg

all: compiled.pdf

compiled.pdf: main.tex
	pdflatex $(basename $<) 
	bibtex $(basename $<) 
	pdflatex $(basename $<) 
	pdflatex $(basename $<) 
	mv $(basename $<).pdf $@

clean: 
	rm -rf $(COMPFILES)
