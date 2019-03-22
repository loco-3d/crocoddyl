# Copyright Projet LAGADIC, 2004
#   http://www.irisa.fr/lagadic
#
#   File: Makefile
#   Author: Nicolas Mansard
#
#   Compilation des sources .tex en .dvi, .ps ...
#
#   Version Control
#     $Id: Makefile,v 1.3 2006/09/08 08:31:30 nmansard Exp $

RM := rm -f
MV := mv
CP := cp -f
AR := ar
TEX2DVI :=  pdflatex -interaction=nonstopmode -output-format dvi
DVI2PS := dvips -Ppdf -G0 -tb4
PS2PDF := ps2pdf -dCompatibilityLevel=1.4 -dMAxSubsetPct=100 -dSubsetFonts=true -dEmbedAllFonts=true -sPAPERSIZE=letter 
TEX2PDF := pdflatex -interaction=nonstopmode
BIBTEX := bibtex
GZIP := gzip -c
XDVI := xdvi -watchfile 0.5 -s 9
TGZ := tar -czvf 

# ---------------------------------------------
# --- REPERTOIRES -----------------------------
# ---------------------------------------------

PROJECT_DIR		= .
ROOT_DIR		= .
IMG_DIR      		= $(ROOT_DIR)
BIB_DIR   		=. 

# ---------------------------------------------
# --- OBJETS ----------------------------------
# ---------------------------------------------

OBJS	=	feasible_ddp
OBJS2	=	

BIB	=	biblio.bib

OBJS_TEX	=	$(OBJS:%=%.tex)
OBJS_DVI	=	$(OBJS:%=%.dvi)
OBJS_PS 	=	$(OBJS:%=%.ps)
OBJS_PS_GZ	=	$(OBJS:%=%.ps.gz)
OBJS_PDF	=	$(OBJS:%=%.pdf)

IMPRIMANTE	=	aragon # mCjaune0 #

TGZ_FILE        =       ../icra10.tgz

# ---------------------------------------------
# --- REGLES ----------------------------------
# ---------------------------------------------

# --->
# ---> Regle generale
# --->

.PHONY = all 
all: 

dvi: $(OBJS_DVI)

ps: $(OBJS_PS)

psgz: $(OBJS_PS_GZ)

pdf: $(OBJS_PDF)

print: $(OBJS_PS) 
	lpr -P $(IMPRIMANTE) $^
	@touch print

view: pdf
	evince $(OBJS_PDF) &

edit:
	emacs $(OBJS_TEX) $(OBJS2:%=%.tex) &
# --->
# ---> Regles generiques
# --->

%.dvi: %.tex $(OBJS2:%=%.tex) $(BIB)
	$(TEX2DVI) $< 

bib: 
	$(TEX2PDF) $(OBJS_TEX) 
	$(BIBTEX) $(OBJS)
	$(TEX2PDF) $(OBJS_TEX) 
	$(TEX2PDF) $(OBJS_TEX)

%.ps: %.dvi
	$(DVI2PS) $< -o $@

%.pdf: %.tex $(OBJS2:%=%.tex)
	$(TEX2PDF) $< $@

%.ps.gz: %.ps
	$(GZIP) $< > $@


# --->
# --->
# ---> Regles de nettoyage
# --->

.PHONY = clean
clean: 
	$(RM) *.aux *.blg *.dvi *.log *.bbl
	$(RM) *~ 
	$(RM) core 
	$(RM) print
	$(RM) -r auto

.PHONY = clean_all
clean_all: clean 
	$(RM) *.ps *.pdf

.PHONY = tgz
tgz: $(TGZ_FILE)

$(TGZ_FILE): clean_all
	@echo "Creation de $@"
	$(TGZ) $(TGZ_FILE) .
