FILES=Makefile non_blocking.c non_blocking.h stack.c stack.h stack_test.c compile run variables drawplots.r Questions/questions.pdf
ARCHIVE=Lab2.zip

NB_THREADS=3
NON_BLOCKING=0
SUFFIX=
MEASURE=0
MAX_PUSH_POP=5000
OUT=stack$(SUFFIX)

STACK_SUFFIX=-$(NON_BLOCKING)
NON_BLOCKING_SUFFIX=-$(NON_BLOCKING)
## Forces assert() calls to be disabled when running measurements or when NDEBUG is defined to non-zero value. If DEBUG is defined to non-zero, tforward it and don't define NDEBUG unless MEASURE is also non-zero in which case DEBUG is not forwarded and NDEBUG is set to 1
NDEBUG_FLAG=$(if $(patsubst 0,,$(MEASURE)),-DNDEBUG=1,$(if $(DEBUG),$(if $(patsubst 0,,$(DEBUG)),-DDEBUG=$(DEBUG),$(if $(NDEBUG),$(if $(patsubst 0,,$(NDEBUG)),-DNDEBUG=$(NDEBUG),-DDEBUG=$(DEBUG)),-DDEBUG=$(DEBUG))),$(if $(NDEBUG),$(if $(patsubst 0,,$(NDEBUG)),-DNDEBUG=1,),)))

CFLAGS=-g -O0 -Wall -pthread -lrt -DNB_THREADS=$(NB_THREADS) -DNON_BLOCKING=$(NON_BLOCKING) -DMEASURE=$(MEASURE) -DMAX_PUSH_POP=$(MAX_PUSH_POP) $(NDEBUG_FLAG)

all: $(OUT)

clean:
	$(RM) stack
	$(RM) stack-*
	$(RM) *.o
	
$(OUT): stack_test.c stack$(STACK_SUFFIX).o nonblocking$(NON_BLOCKING_SUFFIX).o
	gcc $(CFLAGS) stack$(STACK_SUFFIX).o non_blocking$(NON_BLOCKING_SUFFIX).o stack_test.c -o $(OUT)

stack$(STACK_SUFFIX).o: stack.c stack.h
	gcc $(CFLAGS) -c -o stack$(STACK_SUFFIX).o stack.c
	
nonblocking$(NON_BLOCKING_SUFFIX).o: non_blocking.c
	gcc $(CFLAGS) -c -o non_blocking$(NON_BLOCKING_SUFFIX).o non_blocking.c

dist:
	zip $(ARCHIVE) $(FILES)
